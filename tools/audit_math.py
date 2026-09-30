"""Mathematical probes and dense-MAC accounting for DSOR v3.1/v3.2.

Usage: python tools/audit_math.py --repo /path/to/DSOR --output evidence.json
Requires torch, numpy, torchvision, and their dependencies. No data download.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.benchmark import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('evidence.json'))
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
import dsorn_v31 as d
from dsorn_v32 import DSORNetV32Distribution
from deployment import prepare_for_inference
from imaging import NanoImagingModel, unpatchify_4x4

torch.set_num_threads(2)
torch.manual_seed(8401)
evidence = {'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip(),
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'platform': platform.platform(), 'cpu_threads': 2,
                        'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()},
            'source_sha256': {name: hashlib.sha256((args.repo/name).read_bytes()).hexdigest()
                              for name in ('dsorn_v31.py', 'dsorn_v32.py', 'deployment.py', 'imaging.py')}}

def record(name, value):
    evidence[name] = value
    print(name + ': ' + json.dumps(value), flush=True)

# Patch order and exact convolutional representation.
x = torch.arange(2*3*32*32, dtype=torch.float64).reshape(2, 3, 32, 32)
record('patch_roundtrip_max_error', float((unpatchify_4x4(d.patchify_4x4(x))-x).abs().max()))
model = d.DSORNetV31Sequential(10).double().eval()
xc = torch.randn(2,3,32,32,dtype=torch.float64)
stem_conv = F.conv2d(xc, model.patch_embed.weight.reshape(32,3,4,4), model.patch_embed.bias, stride=4)
stem_linear = model.patch_embed(d.patchify_4x4(xc))
record('patch_embed_convolution_max_error', float((stem_conv.flatten(2).transpose(1,2)-stem_linear).abs().max().detach()))
zc = torch.randn(2,64,32,dtype=torch.float64)
merge_conv = F.conv2d(zc.transpose(1,2).reshape(2,32,8,8),
                     model.merge1.weight.reshape(48,2,2,32).permute(0,3,1,2),
                     model.merge1.bias, stride=2)
merge_linear = model.merge1(d.merge_2x2_tokens(zc,8,8))
record('merge_convolution_max_error', float((merge_conv.flatten(2).transpose(1,2)-merge_linear).abs().max().detach()))

# Compare interpolation values and token/coordinate gradients, including boundaries.
sampler_cases = []
for h,w in [(8,8),(4,4),(1,4),(4,1),(1,1)]:
    tokens = torch.randn(2,h*w,5,dtype=torch.float64,requires_grad=True)
    coords = (torch.rand(2,h*w,3,2,dtype=torch.float64)*2.6-1.3)
    probes = torch.tensor([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.],[-1.,-1.],[1.,1.]],dtype=torch.float64)
    coords.reshape(-1,2)[:min(len(probes),coords.numel()//2)] = probes[:min(len(probes),coords.numel()//2)]
    coords.requires_grad_()
    native = d.bilinear_sample_tokens(tokens,coords,h,w,'grid')
    gather = d.bilinear_sample_tokens(tokens,coords,h,w,'gather')
    scalar_probe = torch.randn_like(native)
    ga = torch.autograd.grad((native*scalar_probe).sum(),(tokens,coords),retain_graph=True)
    gb = torch.autograd.grad((gather*scalar_probe).sum(),(tokens,coords))
    row = {'shape':[h,w], 'forward_error':float((native-gather).abs().max()),
           'token_gradient_error':float((ga[0]-gb[0]).abs().max()),
           'coordinate_gradient_error':float((ga[1]-gb[1]).abs().max())}
    assert max(row[k] for k in ('forward_error','token_gradient_error','coordinate_gradient_error'))<1e-10
    sampler_cases.append(row)
record('sampler_equivalence',sampler_cases)
tokens = torch.randn(1,4,2,dtype=torch.float64,requires_grad=True)
coords = torch.tensor([[[[-.63,-.41],[.18,.33]]]*4],dtype=torch.float64,requires_grad=True)
record('sampler_gradcheck', torch.autograd.gradcheck(lambda a,b:d.bilinear_sample_tokens(a,b,2,2,'gather'),
                                                    (tokens,coords),eps=1e-6,atol=1e-5,rtol=1e-4))

# Correct 7/6 transport under clipping, with an independently computed group mean.
p = d.base_grid(8,8,dtype=torch.float64)
wgt = torch.full((1,64,4),.25,dtype=torch.float64)
zero_coords = p.unsqueeze(2).expand(1,64,4,2).clone()
zero_prior,zero_stats = d.aggregate_stage1_trajectory({'coords':zero_coords,'offsets':torch.zeros_like(zero_coords),
                                                     'weights':wgt,'reference':p})
record('zero_motion', {'max_prior':float(zero_prior.abs().max()), 'spread':float(zero_stats[0,0,1]),
                      'expected_spread':math.sqrt(2)/7})
delta = torch.tensor([.03,-.02],dtype=torch.float64)
clipped = (zero_coords+delta).clamp(-1,1)
prior,stats = d.aggregate_stage1_trajectory({'coords':clipped,'offsets':torch.zeros_like(clipped)+delta,
                                          'weights':wgt,'reference':p})
absolute_means = clipped.mean(2).reshape(1,8,8,2).permute(0,3,1,2)
expected = F.avg_pool2d(absolute_means,2,2).flatten(2).transpose(1,2)*(7/6)-d.base_grid(4,4,dtype=torch.float64)
record('clipped_transport_max_error',float((prior-expected).abs().max()))
assert torch.allclose(prior,expected,atol=1e-14,rtol=1e-14)
group_centers = F.avg_pool2d(p.reshape(1,8,8,2).permute(0,3,1,2),2,2)
collapsed = group_centers.repeat_interleave(2,2).repeat_interleave(2,3).permute(0,2,3,1).reshape(1,64,1,2).expand(1,64,4,2).clone().requires_grad_()
collapse_prior,collapse_stats = d.aggregate_stage1_trajectory({'coords':collapsed,'offsets':collapsed-p.unsqueeze(2),
                                                            'weights':wgt,'reference':p})
(collapse_prior.square().sum()+collapse_stats.sum()).backward()
record('collapsed_trajectory_finite_gradient',bool(torch.isfinite(collapsed.grad).all()))

# Different within-token sampling distributions produce the same inherited state.
p2 = d.base_grid(8,8,dtype=torch.float64)
duo = p2.unsqueeze(2).expand(1,64,2,2).clone()
sym = duo.clone()
inner = (p2[0].abs()<.9).all(-1)
sym[0,inner,0,0]-=.1
sym[0,inner,1,0]+=.1
weights2=torch.full((1,64,2),.5,dtype=torch.float64)
def aggregate(coords):
    return d.aggregate_stage1_trajectory({'coords':coords,'offsets':coords-p2.unsqueeze(2),
                                          'weights':weights2,'reference':p2})
pa,sa=aggregate(duo)
pb,sb=aggregate(sym)
record('lost_multimodal_information',{'prior_error':float((pa-pb).abs().max()),
                                    'stats_error':float((sa-sb).abs().max()),
                                    'actual_mean_within_token_variance_a':0.,
                                    'actual_mean_within_token_variance_b':float((sym-duo).square().sum(-1).mean())})

# The historical geometry and entropy defects are fixed on their repro cases.
updater=d.TrajectoryStateUpdate(48).double()
ref=d.base_grid(4,4,dtype=torch.float64)
raw=torch.full((1,16,5,2),.4,dtype=torch.float64,requires_grad=True)
coords4=(ref.unsqueeze(2)+raw).clamp(-1,1)
uniform=torch.full((1,16,5),.2,dtype=torch.float64)
old=torch.zeros(1,16,2,dtype=torch.float64,requires_grad=True)
old_stats=torch.randn(1,16,4,dtype=torch.float64,requires_grad=True)
prior4,stats4,aux4=updater(torch.randn(1,16,48,dtype=torch.float64),old,old_stats,
                         {'coords':coords4,'offsets':raw,'residual':raw,'weights':uniform,'reference':ref})
record('clipped_corner_motion',{'realized':aux4['realized'][0,-1].detach().tolist(),
                               'next_prior':prior4[0,-1].detach().tolist()})
old_stats_grad=torch.autograd.grad(prior4.sum()+stats4.sum(),old_stats,allow_unused=True)[0]
record('old_stats_unused',old_stats_grad is None)
record('coordinatewise_convex_update',bool(((prior4>=torch.minimum(old,aux4['realized'])-1e-15)&
                                          (prior4<=torch.maximum(old,aux4['realized'])+1e-15)).all()))
collapse=torch.zeros(1,16,5,2,dtype=torch.float64,requires_grad=True)
cp,cs,_=updater(torch.zeros(1,16,48,dtype=torch.float64),torch.zeros(1,16,2,dtype=torch.float64),
               torch.zeros(1,16,4,dtype=torch.float64),{'coords':ref.unsqueeze(2)+collapse,'offsets':collapse,
                'residual':collapse,'weights':uniform,'reference':ref})
(cp.sum()+cs.sum()).backward()
record('zero_spread_state_finite_gradient',bool(torch.isfinite(collapse.grad).all()))
entropy_cases=[]
for dtype in (torch.float64,torch.float32,torch.float16,torch.bfloat16):
    weights=torch.tensor([[1.,0.],[.5,.5]],dtype=dtype,requires_grad=True)
    ent=d.normalized_entropy(weights)
    ent.sum().backward()
    single=d.normalized_entropy(torch.ones(1,1,dtype=dtype))
    assert torch.isfinite(ent).all() and torch.isfinite(weights.grad).all() and single.item()==0.
    entropy_cases.append({'dtype':str(dtype),'entropy':ent.detach().tolist(),'finite_gradients':True})
record('entropy_edge_cases',entropy_cases)
half=torch.tensor([1e-4],dtype=torch.float16,requires_grad=True)
spread=d.zero_safe_sqrt(half.square())
spread.sum().backward()
record('raw_fp16_variance_underflow',{'true_length':1e-4,'computed_spread':float(spread.detach()[0]),
                                 'computed_gradient':float(half.grad[0])})
v0=torch.tensor([1.,0.])
v1=torch.tensor([0.,1.])
eta=torch.tensor([1e-4,1.-1e-4])
record('coordinate_gate_norm_counterexample',{'old_norm':float(v0.norm()),'realized_norm':float(v1.norm()),
                                             'next_norm':float((v0+eta*(v1-v0)).norm())})

# The regularizer controls expected residual drift rather than residual energy.
res=torch.tensor([[[[.2,0.],[-.2,0.]]]])
reg_aux={'stage2':[{'residual':res,'weights':torch.tensor([[[.5,.5]]])}]}
record('regularizer_cancellation',{'regularizer':float(d.trajectory_regularizer(reg_aux)),
                                 'mean_squared_residual_norm':float(res.square().sum(-1).mean())})

# Full-network efficient/unoptimized algebra, analytical gradients, and deployment.
comparison=[]
for dtype in (torch.float64,torch.float32):
    efficient=d.DSORNetV31Sequential(10).to(dtype).eval()
    ordinary=copy.deepcopy(efficient)
    for module in ordinary.modules():
        if isinstance(module,(d.IndependentRouter,d.InheritedRouter)):module.efficient=False
    input_a=torch.randn(2,3,32,32,dtype=dtype,requires_grad=True)
    input_b=input_a.detach().clone().requires_grad_()
    ya,aa=efficient(input_a,True)
    yb,ab=ordinary(input_b,True)
    probe=torch.randn_like(ya)
    la=(ya*probe).sum()+.01*d.trajectory_regularizer(aa)
    lb=(yb*probe).sum()+.01*d.trajectory_regularizer(ab)
    la.backward();lb.backward()
    param_error=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(efficient.parameters(),ordinary.parameters()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in efficient.parameters())
    prepared=prepare_for_inference(efficient)
    frozen=prepared(input_a.detach())
    comparison.append({'dtype':str(dtype),'logit_error':float((ya-yb).abs().max().detach()),
                       'input_gradient_error':float((input_a.grad-input_b.grad).abs().max()),
                       'parameter_gradient_error':param_error,
                       'deployment_logit_error':float((ya.detach()-frozen).abs().max()),
                       'all_parameter_gradients_finite':True})
record('full_network_equivalence',comparison)
amp=d.DSORNetV31Sequential(10)
try:
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out,aux=amp(torch.randn(2,3,32,32),True)
        loss=F.cross_entropy(out,torch.tensor([0,1]))+.01*d.trajectory_regularizer(aux)
    loss.backward()
    record('cpu_bfloat16_autocast',{'finite_loss':bool(torch.isfinite(loss)),
                                   'finite_gradients':all(p.grad is None or torch.isfinite(p.grad).all() for p in amp.parameters())})
except (RuntimeError,TypeError) as exc:
    record('cpu_bfloat16_autocast',{'error':str(exc)})
img=NanoImagingModel().eval()
raw_rgb=torch.rand(2,3,32,32)
record('restoration_identity_error',float((img(raw_rgb,task='restore')-raw_rgb).abs().max().detach()))

# Count every dense matrix multiply, including functional projections.
class MacCounter(TorchDispatchMode):
    def __init__(self):super().__init__();self.total=0;self.rows=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if str(func) == 'aten.linear.default':
            a,b=args[:2];cost=a.numel()*b.shape[0]
        elif func is torch.ops.aten.mm.default:
            a,b=args[:2];cost=a.shape[0]*a.shape[1]*b.shape[1]
        elif func is torch.ops.aten.addmm.default:
            a,b=args[1:3];cost=a.shape[0]*a.shape[1]*b.shape[1]
        elif func is torch.ops.aten.bmm.default:
            a,b=args[:2];cost=a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        else:cost=0
        self.total+=int(cost)
        if cost:self.rows.append({'op':str(func),'macs':int(cost)})
        return func(*args,**(kwargs or {}))

canonical=d.DSORNetV31Sequential(10).eval()
unoptimized=copy.deepcopy(canonical)
for module in unoptimized.modules():
    if isinstance(module,(d.IndependentRouter,d.InheritedRouter)):module.efficient=False
frozen=prepare_for_inference(canonical)
v2=d.DSORNetV2Independent(10).eval()
v32=DSORNetV32Distribution(10).eval()
frozen_v32=prepare_for_inference(v32)
macs=[]
for batch in (1,2,128):
    for name,net in [('v31_unoptimized',unoptimized),('v31_canonical',canonical),('v31_prepared',frozen),('v2',v2),('v32',v32),('v32_prepared',frozen_v32)]:
        counter=MacCounter()
        with torch.inference_mode(),counter:net(torch.randn(batch,3,32,32))
        macs.append({'model':name,'batch':batch,'dense_macs':counter.total,'dense_macs_per_image':counter.total/batch})
record('dense_matrix_macs',macs)
record('parameters',{cls.__name__:sum(p.numel() for p in cls(num_classes=10).parameters())
                     for cls in (d.DSORNetV31Sequential,d.DSORNetV2Independent,DSORNetV32Distribution,NanoImagingModel)})
bench=[]
with torch.inference_mode():
    for batch in (1,32,128):
        xb=torch.randn(batch,3,32,32)
        for name,net in [('unoptimized',unoptimized),('canonical',canonical),('prepared',frozen),('v32',v32),('v32_prepared',frozen_v32)]:
            net(xb)
            result=Timer('net(xb)',globals={'net':net,'xb':xb},num_threads=2).blocked_autorange(min_run_time=.15)
            bench.append({'model':name,'batch':batch,'median_ms':result.median*1000,
                          'iqr_ms':result.iqr*1000,'measurement_blocks':len(result.raw_times)})
record('cpu_inference_microbenchmark',bench)

assert evidence['patch_roundtrip_max_error']==0.
assert evidence['patch_embed_convolution_max_error']<1e-12
assert evidence['merge_convolution_max_error']<1e-12
assert evidence['collapsed_trajectory_finite_gradient'] and evidence['zero_spread_state_finite_gradient']
assert evidence['lost_multimodal_information']['prior_error']<1e-14
assert evidence['lost_multimodal_information']['stats_error']<1e-14
expected={('v31_canonical',1):2685440,('v31_canonical',2):5116416,('v31_prepared',1):2152448,
          ('v31_unoptimized',1):3668480,('v2',1):2298368,
          ('v32',1):2664320,('v32_prepared',1):2131328}
for row in macs:
    key=(row['model'],row['batch'])
    if key in expected:assert row['dense_macs']==expected[key],(key,row['dense_macs'],expected[key])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
print('Evidence written:',args.output,flush=True)
