# DSOR mathematical and conceptual audit

Scope: the public v3.1 code at `24e6fb7`, followed by the experimental v3.2 and
runner changes in this branch. Evidence is in `docs/evidence/math_evidence.json`
and `docs/evidence/pilot_results.json`. The latter's exact measured training
source is commit `1fb94df122bcb9316d6316841dc5e2f6539affbc`; all three source-file
hashes were checked against the original experiment manifests.

## Conclusion

The v3.1 bilinear sampler, 7/6 frame transport, and attention reassociation are
mathematically consistent. Its inherited state discards within-query spatial
modes; its vector gate has weaker guarantees than a convex vector update; its
historical residual penalty can cancel opposing samples. Neither v3.1 nor v3.2
has demonstrated new literature-level novelty, calibrated uncertainty, dynamical
contraction, or state-of-the-art accuracy. v3.2 makes distribution transport and
state mixing explicit and testable. It does not establish an accuracy gain.

At `24e6fb7`, clipping-aware motion, finite zero-variance gradients, and the
one-sample entropy case were already fixed. The earlier audit in this file
referred to an older archive and unavailable `work/math_audit` paths. Those
historical defects must not be attributed to this inspected commit.

## Exact sampler and routing algebra

For a query i and sample m, the actual sampled coordinate is

$$s_{im}=\operatorname{clip}(p_i+\Delta_{im},-1,1).$$

Border-padded bilinear interpolation expresses each sample as

$$u_{im}=\sum_j\beta_j(s_{im})z_j,\qquad \beta_j\ge0,\quad\sum_j\beta_j=1.$$

Softmax produces routing probabilities alpha. Conditional on the realized
coordinates and probabilities, the spatial operator is

$$T_{ij}=\sum_m\alpha_{im}\beta_j(s_{im}),\qquad T_{ij}\ge0,\quad\sum_jT_{ij}=1.$$

Each sample touches at most four grid sites. This conditional spatial mixture
is row-stochastic. Channel projections, LayerNorm, residual paths, position
embeddings, and feed-forward blocks mean the entire network is not a Markov
operator or a globally contractive averaging map. Global mean context affects
all routing decisions, so sparse spatial reads do not imply purely local
functional dependence.

With column vectors and bias-free Q/K/V/P matrices,

$$ (Qz_i)^T(Ku_{im})=(K^TQz_i)^Tu_{im},$$
$$P\sum_m\alpha_{im}Vu_{im}=PV\left(\sum_m\alpha_{im}u_{im}\right).$$

The canonical code projects K's transpose once per query and applies V after
weighted pooling. Frozen inference additionally precomputes K^T Q and PV and
caches fixed-grid positional MLP outputs. No rank truncation or dropped samples
is involved. Floating-point reassociation can change rounded outputs/gradients.
The frozen model is a deep copy with a separate deployment state schema; save
the canonical checkpoint for training and prepare after choosing device/dtype.

Checks on the inspected snapshot: float64 whole-network logit error 2.78e-16,
parameter-gradient error 1.89e-15; float32 errors 1.19e-7 and 7.75e-7.
Native gather and grid_sample values/gradients agreed within 7.11e-15 in float64,
including border equality and grids with a singleton axis. Interior gradcheck
passed. Derivatives at cell boundaries follow PyTorch's selected convention.

## Cross-scale geometry

For fine width n, adjacent fine centers have group mean

$$\bar p_j={4j-n+2\over n-1},\qquad p_j^{coarse}={4j-n+2\over n-2}.$$

Thus the frame conversion is A=(n-1)/(n-2), which is 7/6 for 8→4. v3.1 correctly
averages expected *absolute* locations, converts the frame, then subtracts the
coarse reference. Averaging relative offsets alone misses the reference-frame
change. Expected realized displacement is based on clipped coordinates, not
the raw offset. At corner p=(1,1), an outward proposal (0.4,0.4) produces zero
motion; the current v3.1 correctly reports zero.

The frame conversion is affine, not a sampler call. Transported fine particles
may lie outside [-1,1] in the coarse frame. Clipping them during transport shifts
boundary group means even with zero fine motion. v3.2 keeps these coordinates
and clamps only actual coarse reads. The zero-motion transported group has
mean p_coarse and covariance diag(1/36,1/36), from its four geometric centers.

## Information loss and gate semantics in v3.1

The fine state includes the group-average query centroid and four statistics.
Its spread measures variation *between query centroids*. It omits variation
within each query's sampling distribution. The total-covariance identity is

$$\operatorname{Cov}(S)=E_i[\operatorname{Cov}(S\mid i)]
 +\operatorname{Cov}_i(E[S\mid i]).$$

A reproducible counterexample replaces coincident, equally weighted samples
with symmetric separated samples at interior queries. The inherited prior and
all four statistics are exactly unchanged while mean within-query variance
changes from 0 to 0.005625. Routing-weight entropy can also stay unchanged when
sample positions move or coincide. It is not geometric uncertainty or model
confidence. Zero-motion v3.1 spread is sqrt(2)/7≈0.20203 from grid spacing alone.

The v3.1 state update is coordinatewise:

$$\pi'=\pi+\eta\odot(\mu-\pi),\quad\eta\in(0,1)^2.$$

It stays inside the axis-aligned box between pi and mu, not necessarily their
line segment. With pi=(1,0), mu=(0,1), eta≈(0,1), the output approaches (1,1),
whose norm exceeds both endpoint norms. Learned mu depends on the old state and
features, so no dynamical contraction follows. The `old_stats` argument is not
consumed by v3.1's updater. Recurrence is across model depth, with one state
transition in this network; no state persists across images or time.

The old penalty is the squared expected residual. Opposite residuals ±(0.2,0)
with equal weights give penalty zero despite energy 0.04. That penalty is valid
if the purpose is to suppress mean motion; it does not suppress deformation
energy. The new runner uses the same expected squared residual energy for all
architectures, making the comparison explicit and avoiding cancellation. This
changes the historical training objective and must be reported.

## v3.2 distribution memory

For each coarse token, the transported state is the full fine-query mixture

$$P_j=\sum_{i\in G_j}\sum_m{\alpha_{im}\over4}\,\delta_{A s_{im}}.$$

It carries 16 particles at the first coarse block (four queries × four samples).
The next router computes five kernel-conditioned centers using

$$r_{mk}\propto w_k\exp[-\|q_m-x_k\|^2/(2\sigma_m^2)],\qquad
 c_m=\sum_kr_{mk}x_k.$$

Fixed distinct query anchors plus learned residuals provide multiple reads.
The bandwidth is bounded between 0.05 and 0.60. Features include frame-normalized
mean displacement, centered covariance entries, and boundary excess. These are
sampling geometry, not predictive uncertainty. Kernel reads can still merge
modes, especially at large bandwidth; retained particles do not guarantee every
mode is used or improve the final representation.

A single scalar gate eta per token updates the probability distribution:

$$P'=(1-\eta)P+\eta Q,\qquad0\le\eta\le1.$$

This retains old particles and the five new realized particles, giving 21
particles at the second coarse block. Its mean and covariance obey

$$m'=(1-\eta)m_P+\eta m_Q,$$
$$C'=(1-\eta)C_P+\eta C_Q+
\eta(1-\eta)(m_Q-m_P)(m_Q-m_P)^T.$$

The mean is now a vector convex combination. Covariance is positive
semidefinite in exact arithmetic. Tests cover the full covariance identity,
endpoint gates, normalization, collapsed particles, finite gradients, and equal
mean/different covariance states. Full learned dynamics remain noncontractive
unless additional derivative bounds are established. Probability mass must be
nonnegative with positive total mass; internal softmax/mixture constructors
satisfy this precondition.

Geometry reductions now accumulate in at least float32 (float64 remains
float64), including legacy entropy/spread reductions. This avoids raw float16
squaring underflow: a 1e-4 displacement otherwise squares to zero. It does not
remove coordinate quantization or overflow risks in every mixed-precision
operation. CUDA AMP has a hardware-gated optimizer-step test; GPU execution was
not available in the local audit environment.

The mean-only ablation uses exactly the same initial parameters and collapses
the carried distribution before each coarse read. This isolates carried
higher-order geometry within v3.2 more closely than comparing architectures
with different MLP/state-update layouts.

## Complexity and cost

Let B,N,D,M,K denote batch size, token count, channel width, routed samples and
carried particles. The optimized v3.1 router is

$$\Theta(BND^2+BNMD)$$

for dense projections, routing MLPs and sample-feature operations at fixed M.
Sample-feature storage is O(BNMD); weights/parameters and autograd storage add
to it. There is no N×N attention matrix, but the model is not linear in D.
Frozen inference removes training-time matrix work and fixed-grid MLP calls.

v3.2 adds O(BNMK) kernel responsibilities/distances and O(BNK) moments. Its
current K values are 16 and 21. Unbounded repeated mixture updates would grow
K linearly with depth; summing kernel work over L blocks can become quadratic
in L. Generalizing this prototype to deep stacks needs a fixed-budget memory
policy and accuracy-controlled compression/resampling tests.

Exact dense matrix MACs for CIFAR-10, including functional linear operations:

| Model | Parameters | B=1 dense MACs | General-batch dense MACs |
|---|---:|---:|---:|
| v2 independent | 120,884 | 2,298,368 | 2,043,904 B + 254,464 |
| v3.1 canonical | 145,466 | 2,685,440 | 2,430,976 B + 254,464 |
| v3.1 prepared | canonical source | 2,152,448 | 2,152,448 B |
| v3.2 canonical | 144,121 | 2,664,320 | 2,409,856 B + 254,464 |
| v3.2 prepared | canonical source | 2,131,328 | 2,131,328 B |

These counts exclude interpolation, softmax, reductions, activation functions,
LayerNorm and particle-kernel elementwise arithmetic. The v3.1 q/k/v/output
projection group saves 983,040 MAC/image against the per-sample projection
fallback (63.83% for that group only). Whole-network canonical B=1 dense work
falls from 3,668,480 to 2,685,440. Neither percentage is a wall-time speedup.
v3.2 has 1,345 fewer parameters but incurs extra kernel/moment work; small
noisy CPU profiles indicate higher cost. Measure target GPU time and memory
before claiming efficiency (`python tools/benchmark.py --device cuda`).

At B=128, sampled feature tensors alone occupy about 7.75 MiB in float32.
v3.1 parameters use 581,864 bytes; parameters, gradients and Adam's two moments
are about 2.22 MiB, excluding activations, buffers and allocator overhead.
Training memory and latency must be measured, not inferred from weight size.

## Conceptual assumptions and novelty

There are no Conv modules, but patch embedding is exactly a 4×4 stride-4
convolution and 2×2 token merging is exactly a stride-2 convolution after
reordering weights. Both equivalences were tested with zero numerical error.
Use “no explicit convolution modules” as the implementation description.
Fixed patch alignment, learned coordinates, boundary padding and absolute
position embeddings do not confer arbitrary-pixel translation equivariance.
Learned sample positions are routing choices, not semantic correspondences,
causal explanations, calibrated uncertainty, or physically conserved motion.

Learned offsets and sparse continuous spatial attention are established ideas:

- [Deformable Convolutional Networks (2017)](https://arxiv.org/abs/1703.06211).
- [Deformable DETR (2020/2021)](https://arxiv.org/abs/2010.04159).
- [Vision Transformer with Deformable Attention (2022)](https://arxiv.org/abs/2201.00520).

The defensible proposed contribution is explicit cross-scale probability
transport followed by kernel-conditioned reuse and an exact scalar mixture
state update in a small classifier. This is a research hypothesis, not a claim
that no previous work contains the same combination. A comprehensive prior-art
search and controlled ablations are still required. The pilot does not show
that carrying multiple modes is useful for classification accuracy.

## Accuracy evidence and acceptance criterion

Three paired seeds, 1,000 training images, 500 disjoint validation images,
30 epochs, full 10,000-image CIFAR-10 test, identical augmentation/order/objective:

| Model | Test accuracy, mean ± seed SD | Best-validation mean |
|---|---:|---:|
| v3.1 | 40.22% ± 1.02 pp | 35.67% |
| v3.2 | 40.11% ± 0.30 pp | 35.53% |
| v3.2 mean-only | not evaluated | 35.47% |

Paired test differences (v3.2−v3.1): +1.34, −1.26, −0.39 percentage points;
mean −0.10 pp. An illustrative three-seed t interval spans approximately
−3.39 to +3.18 pp; it depends on a normality assumption and one fixed subset,
so it is not a population/generalization guarantee. Smaller observed SD with
three seeds does not establish greater stability. The subsequent mean-only
experiment used validation only. All results are exploratory, not a full-data
benchmark. v3.1 therefore stays the default and v3.2 remains experimental.

Before promoting v3.2, predeclare a full-training protocol, compare v2/v3.1,
mean-only and full distribution memory, include a standard capacity-matched
vision baseline, and repeat with 3–5+ paired seeds. Select settings using
validation, freeze the protocol, then perform final testing. Report accuracy,
seed differences, latency/memory and ablation results together.

## Reproduction and verification limits

```
python -m pytest -q
python tools/audit_math.py --repo . --output runs/math-evidence.json
python tools/benchmark.py --device cuda --output runs/gpu-benchmark.json
```

Local isolated Python 3.11.16 and 3.12.14 reference environments both passed
31 tests, with one CUDA test skipped because hardware was unavailable.
Dependency checks and torchvision imports passed. CPU checkpoint resume was
bitwise equal to uninterrupted training for model/selected weights,
optimizer, scheduler, scaler and RNG. Notebook schema, unique cell IDs and
all code-cell Python syntax passed; an actual Colab/T4 session was not run.
CUDA grid_sample backward can be nondeterministic, and exact replay is not
promised across devices or software versions. See
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
