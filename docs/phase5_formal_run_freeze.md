# Phase 5 formal-run launch freeze

Status: `PASS - target candidate validated; formal baseline SHA reported in final handoff`

Formal training executed during Phase 5A: **NO**

## A. Identity

- Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Phase 0 freeze: `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8`
- Phase 1 architecture: `99753b8cf2a5457faed8ddb695793a7f38e99729`
- Phase 2 data: `f1ff88b532f4d77803b36e4f48fa1b407e01a1b7`
- Phase 3 engine: `803f09a7a410318a8aea79a428746d63fdc817a7`
- Phase 3 boundary maintenance:
  `ee18d4c2ae54b41fb6f800a42c90c072f754355e`
- Local-environment ignore:
  `c4dd870b328bf8bc3c5d5aed35910935de227447`
- Phase 4 RTX 3070 Ti validation:
  `ef6363e0228fac51f70c77a79ab548e6fc60dc1f`
- Phase 5 formal code baseline: reported in the final handoff because a commit
  cannot contain its own SHA
- Frozen config:
  `configs/wrn16_8_arxiv_v4_frozen.yaml`
- Frozen config SHA-256:
  `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`
- CIFAR-10 archive SHA-256:
  `6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE`

The Phase 4 commit is the validated hardware parent. It is not the formal training
code SHA. The full Phase 5 freeze commit will become the single immutable baseline
for seeds 1 through 5.

## B. Frozen research protocol

| Field | Formal value |
|---|---|
| Model | WRN-16-8, `B(3,3)`, depth 16, widen factor 8 |
| Blocks/groups | 2 blocks; channels 16 / 128 / 256 / 512 |
| Dropout | 0 |
| Trainable parameters | 10,961,370 |
| Normalization | byte scale; mean `(125.3,123.0,113.9)`, std `(63.0,62.1,66.7)` |
| Augmentation | flip, reflection pad 4, random 32x32 crop, normalization |
| Loss | `CrossEntropyLoss(reduction="mean", label_smoothing=0.0)` |
| Optimizer | ordinary `torch.optim.SGD`, one complete trainable-parameter group |
| SGD | LR 0.1, momentum 0.9, dampening 0, coupled decay 0.0005, Nesterov |
| Batch | 128; train drop-last; test includes all samples |
| Train batches/epoch | 390 |
| Presented train samples/epoch | 49,920 |
| Epochs / updates | 200 / 78,000 |

The one-based LR schedule is applied before the first update of each epoch:

| Epochs | LR |
|---|---:|
| 1-59 | 0.1 |
| 60-119 | 0.02 |
| 120-159 | 0.004 |
| 160-200 | 0.0008 |

The Phase 2 epoch-data seed remains exactly the one-based epoch number 1 through
200 and remains independent of formal run seed.

## C. Formal seeds and current authorization

- Frozen formal seed set: `[1, 2, 3, 4, 5]`
- Current human-authorized seed: `1`
- Seeds 2, 3, 4 and 5: **NOT AUTHORIZED**

The launcher source accepts all five frozen seeds without a future source change.
Execution additionally requires the requested seed to equal the independently
provided runtime authorization seed. The current authorization gate permits only
seed 1.

## D. Code-baseline rule

Formal training can run only from the final Phase 5 freeze commit. The launcher
requires a full expected code SHA, verifies `HEAD`, rejects tracked drift and
rejects arbitrary untracked source/config content. The established untracked
`references/` evidence tree is allowed; ignored `.venv` and `data` remain local
infrastructure.

Before the freeze commit, `--validate-freeze-candidate` permits a dry-run only for
the explicitly enumerated Phase 5 candidate files. It additionally requires
`WRN_PHASE5_CANDIDATE_VALIDATION=1` and can never be combined with formal
execution. After commit, this candidate mode is not used: the ordinary clean-tree
dry-run must pass against the exact Phase 5 SHA.

## E. Target environment lock

| Field | Locked value |
|---|---|
| OS | Windows 10, build string `Windows-10-10.0.26100-SP0` |
| Python | 3.11.9, repository `.venv/Scripts/python.exe` |
| PyTorch | 2.13.0+cu126 |
| torchvision | 0.28.0+cu126 |
| NumPy | 2.4.4 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| CUDA build | 12.6 |
| cuDNN | 91002 |
| NVIDIA driver | 591.86 |
| GPU | NVIDIA GeForce RTX 3070 Ti, exactly one device at `cuda:0` |
| VRAM / compute capability | 8,589,410,304 bytes / 8.6 |

The target dry-run recorded pip `26.2.1` and a credential-free complete installed
`name==version` inventory. The reviewed inventory is frozen in
`docs/phase5_environment_lock.txt`.

## F. Numerical policy

- eager FP32 model, inputs and logits;
- CUDA autocast/AMP disabled;
- TF32 disabled for matmul and cuDNN;
- `torch.compile` unused;
- cuDNN benchmark disabled;
- cuDNN deterministic enabled;
- deterministic algorithms enabled;
- deterministic warn-only disabled;
- no fused optimizer, gradient clipping, EMA, SWA, warmup, adaptive schedule,
  mixup, cutmix or label smoothing.

Any policy or deterministic-operation failure stops before training or stops the
run without automatic fallback.

## G. Run manifest schema

Before the first formal CIFAR optimizer step, `run_manifest.json` binds:

- schema, target and deterministic run identity;
- seed, the five-seed set and current authorized seed;
- full Phase 5 code SHA and checkpoint format;
- frozen config path/SHA and every manifest-bound Frozen v1 hash;
- CIFAR-10 identity, archive SHA, official split integrity and lengths;
- model identity/parameter count, data, optimizer, loss, schedule and duration;
- epoch-data policy, checkpoint plan and epoch-200-only result selection;
- exact target environment and a credential-free package inventory;
- effective numerical policy;
- initial model/optimizer/cursor fingerprint;
- external output directory and UTC start time;
- the no-post-seed1-tuning invalidation rule.

The fingerprint reads initialized state without drawing RNG values or changing
training state.

## H. External artifact layout

The operator supplies an explicit absolute output root outside the repository. The
deterministic run directory is:

`<output-root>/WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1/seed_01`

It contains:

- `run_manifest.json`;
- `environment.json`;
- `training_log.jsonl`;
- `run_status.json`;
- `checkpoints/epoch_060.pt`;
- `checkpoints/epoch_120.pt`;
- `checkpoints/epoch_160.pt`;
- `checkpoints/epoch_200.pt`;
- `final_result.json` after successful completion;
- `artifact_hashes.json` after successful completion.

The launcher rejects repository-internal roots and an existing new-run directory.
It has no overwrite mode. Formal files are atomic where replacement is required;
the append-only log is flushed and synced at every event.

## I. Checkpoint and resume policy

Scheduled checkpoints are after epochs 60, 120, 160 and 200 at exact global
updates 23,400, 46,800, 62,400 and 78,000. They bind target, frozen config, formal
seed, code SHA, run ID, model, optimizer/momentum, cursor, LR, CPU/CUDA RNG and
environment metadata.

Resume requires an explicit checkpoint under the same run's `checkpoints/`
directory, the latest formal checkpoint, the same seed, run ID, code SHA,
environment and package inventory. A completed run cannot resume. Resume restores
momentum, cursor, LR and all RNG streams before continuing. It records the resume
event/count and never silently creates a fresh run over an existing directory.

## J. Result-selection policy

There is no diagnostic test evaluation before formal completion. After exactly 200
epochs and 78,000 successful updates, the epoch-200 model is checkpointed and then
evaluated once over all 10,000 CIFAR-10 test samples using `model.eval()` and
inference-only execution.

The result records correct, incorrect, total, accuracy and test error. It does not
select a checkpoint, stop early or influence training. The formal per-run value is
the epoch-200 final-checkpoint test error. Five-seed aggregation is forbidden until
all five authorized runs complete under this identical baseline.

## K. No-post-seed1-tuning rule

**The first successful real CIFAR optimizer step for seed 1 exposes the frozen
protocol to result data. From that instant, no architecture, data, augmentation,
loss, optimizer, schedule, evaluation, package, environment or launcher change may
be made based on seed-1 behavior or performance.**

If a genuine source/protocol defect requires a change after that point, seed 1 is
invalidated. A new formal freeze must be created and every formal seed must restart
under the new single baseline. Seed 1 is not a pilot or tuning run, and proximity
to the paper's 4.27% result is not a technical-validity gate.

## L. Formal launch gate

Default launcher invocation is dry-run only and executes zero CIFAR backward and
zero optimizer steps. Formal execution requires all of:

- explicit `--execute-formal`;
- `WRN_REQUIRE_TARGET_CUDA=1`;
- `WRN_FORMAL_TRAINING_AUTHORIZED=1`;
- `WRN_FORMAL_AUTHORIZED_SEED=1`;
- requested `--seed 1`;
- exact full `--expected-code-sha` equal to clean `HEAD`;
- Frozen v1, dataset, environment, numerical, model, optimizer, cursor and output
  safety gates.

The launcher writes `FORMAL_TRAINING_STARTED` only after the first successful
optimizer step, including the seed, epoch/batch, global-update transition, code SHA
and config SHA. Non-finite state or an invariant failure stops without changing the
recipe or skipping a batch.

## M. Target validation results

The actual RTX 3070 Ti `.venv` returned:

| Gate | Result |
|---|---|
| Phase 5 + Phase 4 CUDA targeted tests | PASS - `54 passed, 0 skipped, 1 warning in 21.25s` |
| Full repository pytest | PASS - `183 passed, 0 skipped, 5 warnings in 46.86s` |
| Phase 5 candidate launcher dry-run | PASS |
| Target CUDA identity / policy | PASS |
| Trainable parameter count | PASS - `10,961,370` |
| Initial training fingerprint | `6551fe91eabfa6e2135f5aa7c74d94d9cb743c0ab2b49fb0ba8e396387b42fc3` |
| CIFAR archive / extracted integrity | PASS |
| Frozen v1 artifact hashes | PASS - every manifest binding matched |
| `compileall` | PASS |
| `git diff --check` | PASS |
| Dry-run CIFAR backward | `0` |
| Dry-run CIFAR optimizer steps | `0` |
| Formal training executed | **NO** |

The warnings are the previously reviewed torchvision CIFAR pickle-load
`VisibleDeprecationWarning` under NumPy 2.4. They did not change integrity or test
outcomes. Candidate dry-run reported the Phase 4 parent SHA, as expected before the
Phase 5 freeze commit; the required post-commit dry-run must report the new clean
Phase 5 baseline SHA.

## N. Phase 5A boundary

The required target tests and actual candidate launcher dry-run were returned and
reviewed before the Phase 5 freeze commit. After the commit, a clean-tree
post-commit dry-run must prove `HEAD` equals the formal baseline before any launch
command is issued.

Phase 5A freeze status:

- formal training executed: **NO**;
- CIFAR backward executed by dry-run/tests: **0**;
- CIFAR optimizer steps executed by dry-run/tests: **0**;
- seed 1 launch command issued: **NO**;
- seeds 2-5 authorized: **NO**.
