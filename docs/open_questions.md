# Open questions after Phase 0 freeze

Status: `FROZEN v1 — no Phase 0 blocker remains for Phase 1 architecture validation`

Human approval on 2026-08-10 resolved the project choices below. The remaining unknowns are retained honestly and do not authorize Phase 2.

## Resolved by explicit human approval

- Formal target: arXiv:1605.07146v4 Table 5, CIFAR-10, WRN-16-8 `B(3,3)`, mean/std, moderate augmentation, dropout 0, 4.27%, rounded 11.0M, median over five runs.
- Framework/lineage: PyTorch port; pinned Torch7 is primary for paper-era semantics; later official PyTorch is architecture cross-check only.
- Architecture: depth 16, `N=2`, `k=8`, widths 16/128/256/512, Torch7 shortcut routing, no post-add ReLU.
- Initialization and BN: explicit values in the frozen specification, with assumption classifications preserved.
- Optimization: SGD, Nesterov, momentum/dampening, coupled decay and complete learnable-parameter scope.
- LR schedule: changes at the start of one-based epochs 60, 120 and 160.
- Iterator: batch 128, train drop-last, test include-all.
- Run protocol: seeds `[1,2,3,4,5]`, epoch-data seeds 1…200, deterministic FP32 policy.
- Evaluation: epoch-200 per-run value; median of exactly five preregistered runs.
- Scope: CIFAR-10 only. No result-dependent tuning or seed/aggregation changes.

## Non-blocking source unknowns

| Question | Current classification | Frozen handling |
|---|---|---|
| What exact five seeds produced arXiv v4 Table 5? | `UNKNOWN` | Use approved project seeds `[1,2,3,4,5]`; never call them paper seeds |
| What exact checkpoint-selection rule produced the paper result? | `UNKNOWN` | Use approved epoch-200 rule; no best-test selection |
| What exact paper-run Torch/nn/optim/Torchnet/cudnn revisions were installed? | `UNKNOWN` | Use documented explicit port semantics; do not claim bitwise historical equivalence |
| What was the exact CUDA and two-worker RNG stream? | `UNKNOWN` | Use deterministic run seeds plus separate epoch data seeds; retain deviation disclosure |
| Can every BMVC GCN+ZCA numeric default and artifact be uniquely reconstructed? | `UNKNOWN / PARTIAL PROVENANCE` | Out of frozen target; preserve historical research trail only |

## Phase 2 blockers — not blockers for Phase 1

- Implement and validate the approved augmentation micro-semantics, including transform order, flip branch, offset domain and reflection boundary.
- Define the concrete dataset artifact integrity procedure without downloading data in Phase 1.
- Implement optimizer, scheduler, RNG/data-loader, checkpoint and evaluation components only after separate human authorization.
- Verify deterministic CUDA support on the actual RTX 3070 Ti environment before any formal run.

## Future result interpretation

- Primary success is frozen-protocol fidelity.
- The five-run median may be compared descriptively with 4.27%; no unsupported tolerance is authorized.
- A mismatch triggers an evidence/deviation audit, not hyperparameter tuning.
- Seeds, checkpoint rule, or aggregation must never be chosen after seeing results.

## Phase gate

Phase 1 permits model architecture, initialization, synthetic tensor tests, exact parameter audit and documentation only. Dataset download, DataLoader, augmentation implementation, optimizer/training code, formal runs, tuning and checkpoints remain forbidden.
