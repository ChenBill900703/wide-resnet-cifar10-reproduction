# Wide ResNet CIFAR-10 reproduction specification

Status: `FROZEN v1 — WRN-16-8 arXiv v4 CIFAR-10`

Formal target name: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`

Human approval date: 2026-08-10. This specification is frozen for the formal reproduction. Source classifications remain unchanged: human approval adopts a value for this project but does not turn an implementation assumption into paper evidence.

# Formal frozen target

| Field | Frozen value | Classification / provenance |
|---|---|---|
| Paper | Wide Residual Networks, Zagoruyko & Komodakis | `PAPER-SPECIFIED` |
| Paper version/result family | arXiv:1605.07146v4, Table 5 | `PAPER-SPECIFIED` |
| Dataset/scope | CIFAR-10 only | Project scope, human-approved |
| Model | WRN-16-8, `B(3,3)` | `PAPER-SPECIFIED` table identity |
| Depth / widen factor | 16 / `k=8` | `PAPER-SPECIFIED` table identity |
| Blocks per group | `N=(16-4)/6=2` | `OFFICIAL-CODE-SPECIFIED` formula, adopted |
| Preprocessing | Mean/std, input scale 0–255 | Paper family + official-code numeric supplement |
| Augmentation | Flip → four-sided edge-excluding reflection pad 4 → random 32×32 crop | Paper family + official-code/historical-dependency-derived semantics, adopted |
| Dropout | 0; no active dropout module | `PAPER-SPECIFIED`, Table 5 no-dropout family |
| Reference error | 4.27% CIFAR-10 test error | `PAPER-SPECIFIED`, arXiv v4 p. 8 Table 5 |
| Paper aggregation | Median over 5 runs | `PAPER-SPECIFIED`, Table 5 caption |
| Project seeds | 1, 2, 3, 4, 5 | `IMPLEMENTATION-ASSUMPTION — APPROVED`; not paper seeds |
| Paper parameter count | 11.0M, rounded | `PAPER-SPECIFIED`, Table 5 |

BMVC WRN-40-4 / GCN+ZCA remains an alternative historical track only. Its ZCA preprocessing, 4.97%/4.81% values and one-time aggregation must never be mixed into this frozen target.

## Source identities

- Local paper: `references/wide_residual_networks.pdf`, arXiv:1605.07146v4, SHA-256 `AF606AEB02AE3D713A3BA7AD34A5A583D911A5FD33E52B76507DC37A8ED021E9`.
- Official repository: `szagoruyko/wide-residual-networks`, commit `ae6d0d0561484172790c7a63c8ce6ade5a5a2914`.
- Paper-era/reference Torch7 remains the primary official-code architecture/training-semantics source.
- Later official PyTorch is architecture cross-check only and cannot overwrite Torch7 initialization, BN, optimizer, Nesterov, RNG, dropout or iterator behavior.
- Date-compatible historical dependency snapshots are evidence candidates, not a recovered paper-run lockfile; exact installed revisions remain unknown.

# Frozen architecture

## Topology

- Stem: 3×3 convolution, 3→16 channels, stride 1, padding 1, no bias; output 32×32.
- Group 1: two `B(3,3)` blocks, 128 channels (`16k`), output 32×32. First block uses a 1×1 projection because channels change, stride 1.
- Group 2: two blocks, 256 channels (`32k`), output 16×16. First block projection stride 2.
- Group 3: two blocks, 512 channels (`64k`), output 8×8. First block projection stride 2.
- Final: BN → ReLU → global average pooling → Linear(512,10).

## Block tensor routing

The frozen port adopts pinned Torch7 `wide_basic` semantics:

1. First residual 3×3 convolution consumes `ReLU(BN(x))`.
2. Dimension-preserving shortcut is raw `x` through identity.
3. Dimension-changing shortcut is a 1×1 projection of `ReLU(BN(x))`, not raw `x`.
4. Projection stride matches stage downsampling.
5. Residual addition has no post-add ReLU.
6. The next block begins its own BN→ReLU.

Phase 1 must prove these facts with controlled tensor-level tests, not module names or shapes alone.

# Frozen preprocessing and augmentation

## Normalization

Input values remain on the 0–255 numerical scale before normalization:

- Mean: R 125.3, G 123.0, B 113.9.
- Standard deviation: R 63.0, G 62.1, B 66.7.

The mean/std family is paper-specified; exact numbers are `OFFICIAL-CODE-SPECIFIED` and human-adopted.

## Augmentation semantics

- Order: horizontal flip → reflection padding → random crop.
- Flip draw: probability 0.5; the historical branch draws 0/1, where 0 flips and 1 leaves input unchanged.
- Padding: 4 pixels left/right/top/bottom; edge-excluding reflection candidate adopted from date-compatible Torch nn source.
- Crop: separate zero-based x/y offsets, each uniform on `{0,…,8}`; output 32×32.

These micro-semantics are official-code/historical-dependency derived, not paper facts. A modern mapping must pass future Phase 2 transform-order, flip-branch, offset-domain and reflection-boundary tests before data use. Phase 1 does not implement data transforms.

# Frozen initialization

| Parameter/state | Frozen initialization | Classification |
|---|---|---|
| Conv weight | Normal mean 0, std `sqrt(2/(kernel_width×kernel_height×input_channels))` | `OFFICIAL-CODE-SPECIFIED` |
| Conv bias | Absent | `OFFICIAL-CODE-SPECIFIED` |
| FC weight | Uniform `[-1/sqrt(input_features), +1/sqrt(input_features)]` | `IMPLEMENTATION-ASSUMPTION — APPROVED`, based on historical dependency candidate |
| FC bias | 0 | `OFFICIAL-CODE-SPECIFIED` |
| BN affine | true | `IMPLEMENTATION-ASSUMPTION — APPROVED`, historical dependency candidate |
| BN gamma | Uniform `[0,1)` | Same approved assumption |
| BN beta | 0 | Same approved assumption |
| BN running mean / variance | 0 / 1 | Same approved assumption |
| BN eps / momentum | `1e-5` / `0.1` | Same approved assumption; momentum weights the new batch statistic by 0.1 |

The port must not import later official PyTorch Kaiming FC initialization as paper-era fact.

# Frozen optimization and duration

## Optimizer

- SGD with coupled L2-style weight decay.
- Nesterov: true.
- Momentum: 0.9.
- Dampening: 0.
- Weight decay: 0.0005.
- Initial learning rate: 0.1.

Frozen weight-decay scope: one scalar coupled decay over the complete learnable parameter set. Included: convolution weights, BN gamma/beta, FC weight and FC bias. Excluded: convolution bias because it is absent; BN running statistics because they are buffers. This is an `OFFICIAL-CODE-DERIVED CANDIDATE / HISTORICAL DEPENDENCY UNKNOWN`, approved as port semantics. Modern BN/bias exclusions are forbidden in the formal config.

## LR schedule

The LR change is applied at the start of each milestone epoch:

| One-based epochs | LR |
|---|---:|
| 1–59 | 0.1 |
| 60–119 | 0.02 |
| 120–159 | 0.004 |
| 160–200 | 0.0008 |

Phase 1 config tests must lock boundaries 59→60, 119→120 and 159→160 to prevent scheduler off-by-one. No optimizer/training-loop implementation is authorized in Phase 1.

## Batch and iterator

- Batch size: 128.
- Train: `drop_last=True`.
- Test: include all 10,000 examples.
- Duration: 200 epochs.
- Derived train batches/epoch: `floor(50,000/128)=390`.
- Derived consumed samples/epoch: `390×128=49,920`.
- Derived total updates: `390×200=78,000`.

The 78,000 value is a cross-check, not a paper-specified iteration count.

# Frozen stochasticity and numerical policy

## Seeds

- Formal run seeds: exactly `[1,2,3,4,5]`.
- Each run seed controls Python, NumPy, torch CPU, torch CUDA and model initialization.
- Paper's exact five seeds remain unknown; project seeds are an approved implementation assumption.
- Data/augmentation epoch seed is separate from run seed.
- Define zero-based epoch index `e0∈{0,…,199}` and human-visible epoch `e=e0+1`; use `epoch_data_seed=e0+1=e`. Thus the first through last epoch data seeds are 1 through 200.
- The modern data stream must be deterministic/replayable and record both run seed and epoch data seed. Historical two-worker stream equivalence is not claimed.

## Determinism

- FP32 training.
- AMP false.
- TF32 false.
- `torch.compile` false.
- cuDNN benchmark false.
- Deterministic algorithms true where supported.
- Any unsupported deterministic operation must be reported; silent fallback is forbidden.

This is `IMPLEMENTATION-ASSUMPTION — APPROVED`, chosen for modern single-GPU replayability and explicitly different from Torch7's benchmark/nondeterministic default.

# Frozen checkpoint and evaluation protocol

- Evaluation may run each epoch for diagnostics, but cannot select the official result.
- Save at least epochs 60, 120, 160 and 200.
- Per-run official value: epoch-200 final-checkpoint CIFAR-10 test error.
- No best-test checkpoint, test-driven early stopping, post-hoc selection or aggregation change.
- Run exactly the five preregistered seeds; do not add seeds and select five.
- Formal result: median of the five epoch-200 test errors.
- Individual runs, mean, standard deviation and min/max may be supplementary; paper comparison primary remains the median.

The epoch-200 policy and exact seed set are approved implementation assumptions because the paper's checkpoint rule and seed identities are unknown.

# Frozen success criterion and parameter audit

Primary success: architecture, initialization, preprocessing, optimizer scope, schedule and evaluation protocol pass all frozen-spec fidelity checks.

Secondary comparison: transparently compare the preregistered five-run median against paper reference 4.27%. No unsupported ±X threshold is invented. Failure to reach 4.27% triggers a deviation audit, not hyperparameter tuning or post-hoc config changes.

Required claim wording: “Protocol-faithful reproduction with pre-registered modern-framework assumptions, numerically compared against paper 4.27% median.”

Phase 1 must record exact trainable parameter count and cross-check that its decimal-million rounding agrees with paper's 11.0M. It must not change architecture to force a rounded match.

# Non-blocking historical unknowns

- Exact paper seeds.
- Exact paper-run Torch/nn/optim/Torchnet/cudnn revisions and CUDA RNG stream.
- Exact paper checkpoint-selection mechanism.
- Historical multi-worker shuffle/augmentation stream.

Human-approved project policies above replace these unknowns for the formal port without relabeling them as paper evidence.

# Phase status

Phase 0 is frozen. Phase 1 is architecture implementation + validation only. Dataset download, data pipeline, augmentation implementation, optimizer/training loop, training, tuning, runs and checkpoints remain forbidden. Phase 2 has not started.
