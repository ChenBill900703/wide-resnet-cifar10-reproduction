# Implementation assumptions register

Status: `FROZEN v1 — approved assumptions are project policy, not source facts`

Human approval date: 2026-08-10. Approval permits the named project choice; it never upgrades an `IMPLEMENTATION-ASSUMPTION`, historical dependency candidate, or derived value to paper/official-code evidence.

| ID | Frozen project choice | Evidence limit / reason it remains an assumption | Impact | Status |
|---|---|---|---|---|
| IA-01 | Implement the frozen architecture in PyTorch while preserving pinned Torch7 tensor semantics | The paper-era implementation is Torch7; framework equivalence is not paper-specified | Framework defaults must be overridden and tested | **APPROVED** |
| IA-02 | Projection consumes first BN→ReLU output; identity consumes raw `x`; residual consumes preactivation; no post-add ReLU | Routing is `OFFICIAL-CODE-SPECIFIED`, not uniquely paper-specified; choosing it for the port is a project decision | Changes the function even when shapes agree | **APPROVED**; tensor tests required |
| IA-03 | Mean/std values on 0–255 input: mean `[125.3,123.0,113.9]`, std `[63.0,62.1,66.7]` | Mean/std family is paper-specified; exact numbers are official-code-specified | Scale mismatch would invalidate comparison | **APPROVED** |
| IA-04 | Do not use the BMVC GCN+ZCA track for the frozen target | Pipeline provenance is only partial; exact artifact and historical dependency lock remain unresolved | Avoids silently inventing ZCA numerical defaults | **APPROVED target exclusion**; historical track remains open |
| IA-05 | Official per-run checkpoint is epoch 200; intermediate evaluations/checkpoints cannot select the result | Paper states 200 epochs but does not uniquely state checkpoint selection | Prevents best-test leakage | **APPROVED** |
| IA-06 | Run exactly seeds `[1,2,3,4,5]` and report their median | Paper specifies median over five runs but not seed identities | Seed set may affect numeric result | **APPROVED**; seeds are not paper seeds |
| IA-07 | FP32; AMP=false; TF32=false; compile=false on the RTX 3070 Ti target | These modern hardware controls are absent from the paper | Favors auditability over throughput | **APPROVED specific policy**; other hardware adaptations remain prohibited |
| IA-08 | cuDNN benchmark=false and deterministic algorithms=true where supported; unsupported operations must be reported | This intentionally differs from the official Torch7 benchmark/nondeterministic setting | Improves replayability; exact historical floating-point trajectory is not claimed | **APPROVED** |
| IA-09 | Compute exact trainable parameter count from the Phase 1 model and compare its one-decimal-million rounding with 11.0M | Paper gives only a rounded count | Detects topology errors without altering architecture to force a match | **APPROVED** |
| IA-10 | Separate run seed from epoch data seed; with zero-based `e0`, use `epoch_data_seed=e0+1` for seeds 1…200 | Official iterator reseeds by epoch, but exact historical worker/RNG stream cannot be frozen | Deterministic port is reproducible but not bitwise Torch7 stream emulation | **APPROVED** |
| IA-11 | Use later official PyTorch only as an architecture cross-check | It is official but later and differs in initialization/training details | Prevents lineage drift | **APPROVED safeguard** |
| IA-12 | Train `drop_last=True`; test includes all examples; use 390 batches/epoch and 78,000 updates only as derived checks | Skip/include behavior is official-code-specified, not paper-specified | Affects per-epoch samples and total updates | **APPROVED** |
| IA-13 | Coupled weight decay 0.0005 over every learnable parameter: conv weights, BN gamma/beta, FC weight/bias | Flat-vector historical optimizer behavior is an official/dependency-derived candidate; exact installed revisions are unknown | Differs from modern BN/bias exclusion conventions | **APPROVED**; modern exclusions prohibited |
| IA-14 | Future augmentation mapping: flip → edge-excluding reflection pad 4 → crop; p=0.5; offsets 0…8 | Paper gives family; micro-semantics depend on official/historical dependency evidence | Boundary and RNG details can change samples | **APPROVED for future Phase 2**, with tensor tests; not implemented in Phase 1 |
| IA-15 | FC weight uniform `±1/sqrt(in)`; BN affine, gamma uniform `[0,1)`, beta 0, running mean/var 0/1, eps `1e-5`, momentum `0.1` | Based on date-compatible historical Torch dependency candidate, not a recovered paper-run lockfile | Initialization and BN behavior can affect optimization | **APPROVED explicit modern port approximation** |

## Explicit non-assumptions

- arXiv v4 WRN-16-8 Table 5 is the selected target; BMVC WRN-40-4 is not interchangeable with it.
- Official Torch7 default seed 444 is not claimed to be any of the paper's five seeds.
- Later official PyTorch initialization, optimizer, RNG, dropout omission, and BN defaults do not define paper-era behavior.
- Common PyTorch BN/bias no-decay grouping is not permitted in the frozen target.
- 78,000 updates is `DERIVED FROM OFFICIAL-CODE-SPECIFIED`, not a paper iteration count.
- RTX 3070 Ti feasibility is a project/hardware assessment, not a paper fact.
- Exact historical Torch/nn/optim/Torchnet/cudnn revisions and exact augmentation RNG stream remain unknown.

## Change control

Changing any approved item above requires a documented evidence review, explicit human approval, a new frozen-spec version, and a new manifest. Post-result changes are prohibited.
