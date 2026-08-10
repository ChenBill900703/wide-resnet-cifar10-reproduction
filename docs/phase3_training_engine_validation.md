# Phase 3 training-engine semantics validation

Status: `PASS — synthetic/in-memory semantics only; waiting for human review`

## A. Identity

- Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Phase 0 freeze commit: `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8`
- Phase 1 commit: `99753b8cf2a5457faed8ddb695793a7f38e99729`
- Phase 2 commit: `f1ff88b532f4d77803b36e4f48fa1b407e01a1b7`
- Phase 3 commit: reported in the final handoff because a commit cannot contain
  its own final SHA
- Frozen config SHA-256:
  `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`

Phase 2 is human-approved and closed. Phase 3 authorizes only training-engine
semantics validation on synthetic tensors and controlled in-memory synthetic data.

## B. Environment

- Validation date: 2026-08-10 (Asia/Taipei)
- Python: `3.12.13`
- PyTorch: `2.13.0+cpu`
- CUDA runtime/GPU/cuDNN: unavailable in this Work runtime
- Runtime validation: Windows CPU
- Target NVIDIA GeForce RTX 3070 Ti CUDA/cuDNN preflight: `NOT YET VERIFIED`

The engine semantics in this document were validated only in the current CPU Work
runtime. No GPU deterministic-resume claim is made.

## C. Loss semantics and evidence preflight

The pinned official WRN `train.lua` constructs
`nn.CrossEntropyCriterion()` and supplies it to `tnt.OptimEngine`. The
date-compatible historical `torch/nn` snapshot shows that this criterion is
`LogSoftMax` followed by `ClassNLLCriterion`; the latter defaults
`sizeAverage=true`. The modern port therefore uses:

`torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)`

This reduction mapping is **historical-dependency-backed port semantics**, not a
new paper claim. A tensor test matches it exactly against manual log-softmax plus
mean negative log likelihood. No softmax pre-pass, class weights, label smoothing,
target mixing or custom scaling is present.

Pinned locators, accessed 2026-08-10:

- Official WRN commit `ae6d0d0561484172790c7a63c8ce6ade5a5a2914`,
  `train.lua` lines 166–169 and 231–238.
- Historical `torch/nn` commit
  `f76a44871e63ad32285bd32a3f5ffc9419bbc746`,
  `CrossEntropyCriterion.lua` lines 1–28 and `ClassNLLCriterion.lua` lines 4–13,
  35–46.
- Historical Torchnet commit
  `45b3571d4f8e8e8af854f8dbcb36dd31190a7d00`,
  `engine/optimengine.lua` training loop.

The actual paper-run dependency lock remains unknown, as frozen v1 already records.

## D. Optimizer and parameter scope

The formal factory constructs exactly `torch.optim.SGD` with one parameter group:

| Field | Value |
|---|---:|
| initial LR | 0.1 |
| momentum | 0.9 |
| dampening | 0.0 |
| weight decay | 0.0005 |
| Nesterov | true |
| maximize | false |
| foreach | false |
| fused | false |
| differentiable | false |

The group contains the exact identity set of every `model.parameters()` element
whose `requires_grad` is true, with no missing, duplicate or extra tensor. For the
frozen WRN this includes all convolution weights, every BN gamma/beta, FC weight and
FC bias. Buffers and absent convolution biases are excluded. A negative test removes
BN/bias parameters and proves the audit fails.

## E. Analytical SGD semantics

For parameter `p`, raw gradient `g`, scalar decay `wd`, momentum `m=0.9`,
dampening `0` and LR `lr`, the independently implemented equation is:

1. `d = g + wd*p`
2. first step `b = d`; later `b = m*b_previous + d`
3. Nesterov direction `u = d + m*b`
4. `p = p - lr*u`

Float64 two-step probe with `p0=[1.25,-0.75]`:

| Step | Raw gradient | Momentum buffer after step | Parameter after step |
|---|---|---|---|
| 1 | `[0.4,-0.2]` | `[0.400625,-0.200375]` | `[1.17388125,-0.71192875]` |
| 2 | `[-0.1,0.3]` | `[0.261149440625,0.119306535625]` | `[1.16031910628125,-0.75263074176875]` |

Both steps match ordinary `torch.optim.SGD` within float64 tolerance. A separate
test distinguishes this coupled gradient addition from a decoupled parameter
shrink. Historical equation locator: `torch/optim` commit
`89ef52a03b1c39c645d96023b8748ef84973d4f6`, `sgd.lua` lines 27–73.

## F. Actual WRN synthetic update

A synthetic batch of two 32×32 RGB tensors performs WRN forward, mean cross
entropy and backward. Every trainable parameter has a present finite gradient.
Before the first optimizer step, raw gradients and parameter values are captured for
one scalar from each of:

- convolution weight;
- BN gamma;
- BN beta;
- FC weight;
- FC bias.

Each post-step scalar matches the manual first-step coupled-decay/Nesterov equation.
This directly validates that BN affine parameters and FC bias receive the frozen
coupled weight decay. The input is synthetic; CIFAR-10 is not used.

## G. LR schedule and derived update ranges

LR is applied explicitly at the start of the one-based epoch:

| Epochs | LR | Derived completed-update range using this LR |
|---|---:|---:|
| 1–59 | 0.1 | 1–23,010 |
| 60–119 | 0.02 | 23,011–46,410 |
| 120–159 | 0.004 | 46,411–62,010 |
| 160–200 | 0.0008 | 62,011–78,000 |

Boundary tests cover 1, 2, 59, 60, 61, 119, 120, 121, 159, 160, 161 and 200;
epochs 0 and 201 are rejected. The global-update ranges are derived cross-checks;
the formal source remains the epoch schedule.

## H. Update accounting and cursor

`global_update` means the number of successfully completed `optimizer.step()`
calls. It starts at 0 and advances exactly once after a successful step. Forward,
non-finite loss, backward/non-finite gradient or optimizer failure cannot advance it.

The cursor stores `current_epoch_1`, `next_batch_index_0` and `global_update`, with
the invariant:

`global_update = (current_epoch_1 - 1) * 390 + next_batch_index_0`

For epochs 1–200, the next batch is 0–389. The terminal cursor is exactly epoch 201,
batch 0, update 78,000. A successful 390th update rolls to the next epoch; batch 390
inside an epoch and a 391-step epoch representation are rejected.

Epoch preparation applies LR first and calls the Phase 2 API with zero-based epoch
`epoch_1-1`, yielding data seed `epoch_1`. Integration tests verify epoch/data seed
60 with LR 0.02, 120 with 0.004 and 160 with 0.0008.

## I. Checkpoint schema and policy

The atomic writer saves to a temporary file in the destination directory, flushes
and closes it, then uses `os.replace`. Tests only write under pytest temporary
directories. The schema includes:

- format and engine versions;
- formal target and frozen-config SHA-256;
- model signature and model state dictionary;
- optimizer state dictionary and invariant hyperparameters;
- training cursor and current LR;
- run seed and epoch-data policy identifier;
- Python, NumPy, torch CPU and, when available, torch CUDA RNG states;
- Python and PyTorch versions.

The approved label `epoch N` means after completing epoch N. Tests validate:

| Checkpoint label | Next epoch | Next batch | Global update |
|---|---:|---:|---:|
| 60 | 61 | 0 | 23,400 |
| 120 | 121 | 0 | 46,800 |
| 160 | 161 | 0 | 62,400 |
| 200 | 201 | 0 | 78,000 |

## J. Compatibility and exact resume

Load fails closed for unsupported format, wrong formal target, wrong frozen hash,
model state signature mismatch, optimizer type/hyperparameter mismatch, invalid
cursor, or cursor-inconsistent LR. Model and optimizer are reconstructed and loaded
before checkpoint RNG state is restored.

Controlled deterministic in-memory trajectories validate:

- model state exact equality with `torch.equal`;
- optimizer state and all momentum buffers exact equality;
- cursor, global update and LR equality;
- RNG replay for Python, NumPy and torch CPU;
- interruption after two batches in epoch 1;
- mid-epoch interruption in epoch 3;
- epoch-60 boundary interruption and milestone LR restoration.

No long update loop is used to test milestones; valid cursor simulation is used.

## K. Safety and deterministic controls

Each train step is ordered: train mode, zero gradients, forward logits, mean CE,
finite-loss check, backward, full finite-gradient check, optimizer step, then one
cursor increment. There is no implicit accumulation. Controlled NaN/Inf loss and
gradient tests prove no parameter update and no counter increment. A controlled
optimizer failure also leaves the counter unchanged.

The runtime helper seeds Python, NumPy, torch CPU and available CUDA generators;
disables cuDNN benchmark and TF32; enables deterministic algorithms and cuDNN
deterministic mode. CPU-visible settings and replay pass. Target GPU behavior remains
unverified.

## L. Validation results and boundary

| Validation | Result |
|---|---|
| Phase 3 focused tests | PASS — 74 passed in 3.95s |
| Full repository pytest | PASS — 127 passed in 29.66s |
| Independent non-pytest probe | PASS |
| Phase 1 regression | PASS — 21 passed in 3.68s |
| Phase 2 regression | PASS — 32 passed in 27.50s |
| `compileall` | PASS |
| `git diff --check` | PASS |
| Frozen artifact hash audit | PASS — every manifest-bound artifact matches |

Explicit boundary result:

- **NO CIFAR-10 optimizer step**
- **NO CIFAR-10 training or full train epoch**
- **NO formal or smoke run**
- **NO test-accuracy evaluation or result selection**
- **NO runs/checkpoints artifacts in the repository**
- **NO Phase 4; formal training remains unauthorized**
