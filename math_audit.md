# Independent DSORNet v3.1 mathematical audit

The attached README and original code were examined as task data. No instructions from the archive were treated as authorization. This audit did not change original or improved implementation files.

Reproduction: `python3 work/math_audit/reproduce.py`. Machine-readable output is `work/math_audit/evidence.json`. Tests ran on CPU with PyTorch 2.12.1.

## Confirmed defects

1. **P1: boundary-clipped movement is falsely reported as realized movement.** Original `TrajectoryStateUpdate.forward`, lines 383–401, computes the expected displacement and variance from `router_aux['offsets']`. Both routers clamp `p + offsets` at lines 197 and 285 before sampling. Therefore actual displacement is `coords - p`, not `offsets`. At `p=(1,1)`, five offsets of `(0.4,0.4)` and uniform attention produce actual movement `(0,0)`, but original reported movement is `(0.40000004,0.40000004)` and the initial eta=0.5 update incorrectly sets the next prior to `(0.20000002,0.20000002)`. Fix both realized displacement and its spread using effective offsets. Original stage-1 `expected_off` at line 110 also disagrees with clipped absolute positions and should use effective displacement when its diagnostic is intended to describe actual motion. The defect affects state semantics and changing it intentionally changes model outputs, separate from exact performance optimizations.

2. **P1: zero trajectory variance produces NaN backward gradients.** Original stage-1 spread at lines 133–137 and recurrent spread at lines 398–401 evaluate `sqrt(sum of squares)`. At zero dispersion, the derivative of sqrt is infinite and PyTorch computes `0 * inf` in backward. Five coincident zero offsets yield 160 NaN gradient entries in recurrent state; routing every fine 2x2 cell to its own group center yields 512 NaN entries in stage-1 transport. These stage-1 offsets have maximum component only 0.14285717, inside the model's 0.38 radius. Both forward passes are finite, so a forward-only smoke test misses the issue. Preserve positive forward values exactly while choosing the norm's finite zero subgradient: compute `positive = variance > 0`, `safe = where(positive, variance, ones)`, then `where(positive, sqrt(safe), zeros)`. Merely wrapping an already computed `sqrt(variance)` in `where` does not protect backward. The [official autograd notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html) document masking-after-invalid-operation hazards and non-differentiable-point conventions.

3. **P2: normalized entropy is undefined for one sample.** Lines 126 and 395 divide by `log(M)` with no M=1 case. The public router constructors allow `samples=1`; the resulting deterministic distribution has entropy zero but the code produces NaN. Explicitly return a zero entropy tensor when M=1 and reject M<1.

4. **P2: entropy stabilization underflows in float16.** The `1e-9` added inside the logarithm is zero in float16. Weights `[1,0]` make both entropy and backward gradients nonfinite. A dtype-aware positive floor for the logarithm, or computing entropy in float32 for low precision, prevents `0 * log(0)`. Using a floor alone changes the extremely small positive-weight derivative below the floor; float32 accumulation is preferable if mixed precision support is intended.

## Exact complexity reductions

Let a row query be `q = z Wq^T`, sampled vectors be `s_m`, and bias-free key/value layers have weights `Wk`, `Wv`. Then

`q (s_m Wk^T)^T = (q Wk) s_m^T`,

so `F.linear(q, Wk.T)` can be computed once per token, replacing the key projection once per sample. Also

`sum_m alpha_m (s_m Wv^T) = (sum_m alpha_m s_m) Wv^T`,

so the value projection moves after weighted pooling. These identities preserve the learned parameters, sample count, sampling coordinates, attention weights in exact arithmetic, and all analytical derivatives. They require no rank truncation, quantization, dropped samples, or approximation. Float arithmetic reorders summations, so bitwise equality is not promised. [PyTorch Linear documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) defines the matrix convention underlying these formulas.

Counting multiply-accumulate pairs (MACs), q/k/v/output projections originally cost `(2+2M) B N D²`; reassociation costs `4 B N D²`. The query/sample dot products and weighted pooling remain `O(B N M D)`. Therefore the sample-dependent dense work changes from `O(B N M D²)` to `O(B N D² + B N M D)`; the entire network still has its other dense MLPs and spatial sampling.

| Projection group | Original MAC/image | Optimized MAC/image | Saved |
|---|---:|---:|---:|
| Stage 1, N=64 D=32 M=4 | 655,360 | 262,144 | 393,216 |
| Two stage-2 blocks, N=16 D=48 M=5 | 884,736 | 294,912 | 589,824 |
| Total q/k/v/output projections | 1,540,096 | 557,056 | 983,040 |

This is a 63.83% MAC reduction and 2.765x ratio **for those projections only**. It is not a claim of a 2.765x end-to-end wall-time speedup or a 63.83% reduction in every network operation. Actual performance must be benchmarked at the intended batch size and device, including backward for training.

Independent forward/backward comparison with random sampled tensors and all q/k/v/output weights:

| Dtype and shape | Largest output difference | Largest gradient difference |
|---|---:|---:|
| float64, N64 D32 M4 | 1.78e-15 | 3.55e-14 |
| float64, N16 D48 M5 | 1.78e-15 | 1.24e-14 |
| float32, N64 D32 M4 | 9.54e-7 | 3.81e-5 |
| float32, N16 D48 M5 | 7.15e-7 | 6.68e-6 |

All compared gradients are finite. Gradient errors above use an unnormalized random scalar probe over the entire output, not a batch-mean training loss. They cover gradients with respect to input queries, sampled features, learned score logits, and all four projection weight tensors. Geometry fixes must be disabled or matched in a whole-model equivalence test so intentional correctness changes are not mistaken for reassociation error.

Additional exact opportunities: positional and coordinate embeddings depend only on a fixed grid, so evaluate each on shape `[1,N,2]` and broadcast the result, rather than broadcasting coordinates to B before the MLP. A context MLP's initial affine layer can similarly split local, global, positional, and inherited partitions, computing its globally constant contribution once per image. The latter trades fewer MACs for more/smaller kernels and should be accepted only with a measured runtime benefit.

## Geometry confirmed correct

The 8-to-4 scaling by 7/6 is correct for this coordinate convention. For even fine width n, fine centers are `p_i=-1+2i/(n-1)` and the mean of centers `2j,2j+1` is `(4j-n+2)/(n-1)`. The coarse center is `(4j-n+2)/(n-2)`, hence mean fine center equals `(n-2)/(n-1)` times coarse center. The implementation's span is exactly this factor. [PyTorch grid_sample documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grid_sample.html) confirms `align_corners=True` anchors ±1 at the corner pixel centers and border padding samples the boundary. Constant-motion transport tests that construct coordinates beyond ±1 validate affine transport algebra, but are not valid expected-motion tests for the actual clipping router near borders.

## Checkpoints and experiment cautions

Safely inspected with `torch.load(..., map_location='cpu', weights_only=True)`. The supplied v3.1 checkpoint strictly loads into the original class, has 114 tensors and 145,466 trainable parameters; v2 strictly loads with 92 tensors and 120,884 parameters. All tensors are finite. Checkpoint compatibility confirms shape/schema only, not archived accuracy claims. The archived v3 static model checkpoint contains 108 tensors but its model class is not included in this source file.

The README's 450-example digits pilot is not CIFAR evidence. Small single-seed CIFAR runs can detect runtime or training failures and compare controlled pilot metrics, but cannot establish non-inferiority or improved population accuracy. Tune on a held-out validation subset drawn from training data, and reserve test data for final reporting. Corrections to state semantics can change already-trained checkpoint predictions even though tensor names and shapes remain compatible.

The trajectory regularizer uses the squared norm of the expected residual, not expected squared residual norm. Opposing residuals can cancel. That is a design choice consistent with regularizing the average trajectory and should not be silently changed to a different objective when preserving model fidelity.
