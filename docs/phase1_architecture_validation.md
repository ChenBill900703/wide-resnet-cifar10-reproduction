# Phase 1 architecture validation

Status: `PASS — architecture/init/synthetic validation only`

- Frozen target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Phase 0 freeze commit: `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8`
- Frozen config SHA-256: `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`
- Validation runtime: Python 3.12 bundled workspace runtime; PyTorch `2.13.0+cpu`
- Validation date: 2026-08-10 (Asia/Taipei)

The CPU runtime is used only for Phase 1 structural tests. It is not the future RTX 3070 Ti training environment and no hardware-equivalence claim is made.

## Implemented files

| File | Responsibility |
|---|---|
| `src/wrn/model.py` | Frozen WRN-16-8 topology and Torch7 tensor routing |
| `src/wrn/initialization.py` | Frozen convolution, BN and FC initialization policy |
| `src/wrn/__init__.py` | Explicit architecture exports |
| `tests/test_wrn_architecture.py` | Depth, widths, blocks, projections, stage shapes, dropout absence and synthetic backward |
| `tests/test_wrn_shortcut_semantics.py` | Tensor-level branch-input and post-add behavior |
| `tests/test_wrn_initialization.py` | Initialization formula/state and seed replayability |
| `tests/test_wrn_parameter_count.py` | Exact trainable count, rounded paper cross-check and buffer separation |
| `tests/test_frozen_spec.py` | Frozen identity, LR boundary, epoch seed and iterator-derived checks |

## Architecture mapping

| Frozen requirement | Implemented mapping | Validation |
|---|---|---|
| `N=(16-4)/6=2` | Two `WideBasicBlock` instances in each of three groups | PASS |
| Stem width/shape | 3×3, 3→16, stride 1; 32×32 | PASS |
| Group 1 | 128 channels; two blocks; 32×32 | PASS |
| Group 2 | 256 channels; two blocks; first stride 2; 16×16 | PASS |
| Group 3 | 512 channels; two blocks; first stride 2; 8×8 | PASS |
| Final head | BN→ReLU→global average pool→Linear(512,10) | PASS |
| Dropout 0 | Nonzero dropout rejected; no `nn.Dropout` module | PASS |

Observed shapes for synthetic batch size 2:

| Point | Shape |
|---|---|
| stem | `(2,16,32,32)` |
| group 1 | `(2,128,32,32)` |
| group 2 | `(2,256,16,16)` |
| group 3 | `(2,512,8,8)` |
| logits | `(2,10)` |

## Tensor-level shortcut determination

Controlled hooks and zero-residual tests established:

1. A dimension-changing 1×1 projection consumes the exact first BN→ReLU tensor.
2. The first residual 3×3 convolution consumes that same preactivated tensor.
3. A dimension-preserving identity consumes raw `x`.
4. Negative raw `x` survives a zero-residual block, proving there is no post-add ReLU.
5. Hook order is BN1→ReLU1→conv1→BN2→ReLU2→conv2.

Result: `PASS`; this matches pinned Torch7 semantics and the later official PyTorch architecture cross-check without importing its training defaults.

## Initialization validation

- Every convolution call used normal mean 0 with observed argument `std=sqrt(2/(kW×kH×in_channels))`.
- All convolution biases are absent.
- Every BN uses affine parameters, eps `1e-5`, momentum `0.1`, gamma in `[0,1)`, beta 0, running mean 0 and running variance 1.
- Classifier weight lies within `±1/sqrt(512)` and classifier bias is 0.
- Reconstructing the model after resetting the same torch seed produces exactly equal state dictionaries.

Result: `PASS`. FC/BN policy remains the approved implementation assumption documented in the frozen specification.

## Parameter audit

- Exact trainable parameters: `10,961,370`.
- Decimal millions: `10.96137M`.
- Rounded to one decimal million: `11.0M`.
- Paper cross-check: arXiv v4 Table 5 reports rounded `11.0M`.

Result: `PASS`. The exact count is an implementation audit; only the rounded 11.0M value is paper-specified.

## Validation commands and results

| Command | Result |
|---|---|
| `python -m pytest -p no:cacheprovider -q` with bytecode disabled | `21 passed in 4.90s` |
| `python -m compileall -q src tests` with cache redirected outside repository | PASS |
| Synthetic forward + cross-entropy backward | PASS; finite loss and finite stem/classifier gradients |
| Frozen config parameter-count probe | `10,961,370`, rounded `11.0M` |

## Phase boundary audit

Phase 1 contains no dataset access, dataset download, DataLoader/sampler, augmentation implementation, optimizer, scheduler implementation, training loop, checkpoint, formal run, tuning, or result selection. LR and epoch-seed tests inspect frozen configuration boundaries only. Phase 2 remains unauthorized.
