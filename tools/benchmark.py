"""Measure canonical and prepared inference on the actual target device."""
import argparse
import json
import platform
from pathlib import Path
import sys

import torch
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cifar_experiment import MODELS, resolve_device
from deployment import prepare_for_inference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', default='cpu')
parser.add_argument('--batches', type=int, nargs='+', default=[1, 32, 128])
parser.add_argument('--seconds', type=float, default=1.)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--output', type=Path, default=Path('runs/benchmark.json'))
args = parser.parse_args()
if min(args.batches) < 1 or args.seconds <= 0 or args.threads < 1:
    parser.error('batches, seconds and threads must be positive')
device = resolve_device(args.device)
torch.set_num_threads(args.threads)


@torch.inference_mode()
def run(model, inputs):
    output = model(inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return output


rows = []
for architecture in ('v31', 'v32'):
    torch.manual_seed(17)
    canonical = MODELS[architecture]().to(device).eval()
    prepared = prepare_for_inference(canonical)
    for batch in args.batches:
        inputs = torch.randn(batch, 3, 32, 32, device=device)
        for kind, model in (('canonical', canonical), ('prepared', prepared)):
            for _ in range(5):
                run(model, inputs)
            result = Timer('run(model, inputs)', globals={'run': run, 'model': model, 'inputs': inputs},
                           num_threads=args.threads).blocked_autorange(min_run_time=args.seconds)
            row = {'architecture': architecture, 'kind': kind, 'batch': batch,
                   'median_ms': result.median*1000, 'iqr_ms': result.iqr*1000,
                   'measurement_blocks': len(result.raw_times)}
            if device.type == 'cuda':
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                run(model, inputs)
                row['additional_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)-baseline
            rows.append(row)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'torch': str(torch.__version__), 'python': platform.python_version(),
                                  'device': str(device), 'threads': args.threads,
                                  'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                  'measurements': rows}, indent=2)+'\n')
print(args.output)
