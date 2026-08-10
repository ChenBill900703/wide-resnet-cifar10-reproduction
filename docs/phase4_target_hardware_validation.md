# Phase 4 RTX 3070 Ti target-hardware validation

Status: `PASS - target CUDA environment validated; ready for human formal-training authorization`

## A. Identity

- Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Phase 0 freeze commit: `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8`
- Phase 1 commit: `99753b8cf2a5457faed8ddb695793a7f38e99729`
- Phase 2 commit: `f1ff88b532f4d77803b36e4f48fa1b407e01a1b7`
- Phase 3 commit: `803f09a7a410318a8aea79a428746d63fdc817a7`
- Phase 3 boundary maintenance commit:
  `ee18d4c2ae54b41fb6f800a42c90c072f754355e`
- Local-environment ignore commit: `c4dd870b328bf8bc3c5d5aed35910935de227447`
- Phase 4 commit: reported in the final handoff because a commit cannot contain
  its own final SHA
- Frozen config SHA-256:
  `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`

Phase 4 validates the frozen model and Phase 3 engine on the approved target
hardware. It does not authorize or execute formal CIFAR-10 training.

## B. Execution attribution

The validation harness was implemented and statically audited in the Codex Work
environment, whose PyTorch runtime is CPU-only. Codex did not claim or substitute a
Work-runtime GPU result.

The CUDA commands and tests recorded below were executed by the human operator in
the repository-local Windows `.venv` on the actual RTX 3070 Ti. The complete
terminal outputs were returned to Codex and reviewed before this document was
created.

## C. Actual target environment

Validation date: 2026-08-10 (Asia/Taipei)

| Field | Runtime-reported value |
|---|---|
| OS | `Windows-10-10.0.26100-SP0` |
| Python | `3.11.9` |
| PyTorch | `2.13.0+cu126` |
| torchvision | `0.28.0+cu126` |
| NumPy | `2.4.4` |
| PyYAML | `6.0.3` |
| pytest | `9.1.1` |
| PyTorch CUDA build | `12.6` |
| cuDNN | `91002` |
| NVIDIA driver | `591.86` |
| CUDA available | `true` |
| CUDA device count | `1` |
| selected/current device | `cuda:0` / index `0` |
| GPU | `NVIDIA GeForce RTX 3070 Ti` |
| compute capability | `8.6` |
| total VRAM | `8,589,410,304` bytes |
| default dtype | `torch.float32` |

The identity gate requires the exact GPU name. The VRAM capacity gate uses the
vendor decimal convention (at least 8,000,000,000 bytes), avoiding the false
assumption that an advertised 8 GB device must expose exactly 8 GiB to PyTorch.

## D. Numerical policy

The target process reported and asserted:

| Policy | Effective value |
|---|---|
| FP32 model/input/logits | enabled; all `torch.float32` |
| CUDA autocast / AMP | disabled |
| TF32 | disabled |
| matmul precision API | `fp32_precision = "ieee"` |
| cuDNN precision API | `fp32_precision = "ieee"` |
| `torch.compile` | not used; model has no `_orig_mod` wrapper |
| cuDNN benchmark | disabled |
| cuDNN deterministic | enabled |
| deterministic algorithms | enabled |
| deterministic warn-only | disabled |

The runtime helper uses the installed PyTorch precision API and retains a legacy
`allow_tf32 = False` fallback only for older compatible runtimes. Phase 4 contains
no autocast, GradScaler, AMP or compiled execution path.

## E. CUDA model and synthetic forward/backward

The frozen `WRN-16-8` was constructed directly on explicit `cuda:0`. The target
test recomputed exactly `10,961,370` trainable parameters and required every
parameter, buffer and BatchNorm state tensor to be on `cuda:0`, with no CPU or
mixed-device state.

A synthetic batch of two float32 tensors with shape `2 x 3 x 32 x 32` performed a
training-mode forward, mean cross entropy and backward on CUDA. The tests required
finite float32 logits and loss, plus present, finite CUDA gradients for every
trainable parameter. CIFAR-10 was not accessed by this optimizer test.

## F. CUDA optimizer semantics

The optimizer remained the frozen single-group SGD configuration: LR `0.1`,
momentum `0.9`, dampening `0`, coupled weight decay `0.0005`, and Nesterov enabled.
The full trainable-parameter scope remained inherited from the Phase 3 audited
factory.

On the first synthetic CUDA update, independent expected values used:

1. `g_wd = g + weight_decay * p`
2. first momentum buffer `= g_wd`
3. Nesterov direction `= g_wd + momentum * buffer`
4. updated parameter `= p - lr * direction`

Representative convolution weight, BN gamma, BN beta, FC weight and FC bias values
matched the actual CUDA FP32 update with explicit `rtol=2e-5`, `atol=2e-7`.
Momentum tensors were required to reside on `cuda:0`.

## G. CUDA RNG, checkpoint and exact resume

Low-level CUDA RNG replay saved the `cuda:0` generator state between draws and
required `torch.equal` after restoration. The actual Phase 3 checkpoint path also
saved and restored CUDA RNG state.

Checkpoint tests used only pytest temporary directories. After reconstructing the
model and optimizer, the tests required model parameters, buffers and optimizer
momentum on `cuda:0`, and restored the run seed, cursor, global update, LR and next
CUDA random stream.

The core exact-resume comparison ran synthetic trajectory A uninterrupted for five
updates and trajectory B for two updates, checkpoint/reconstruction, then three
updates. Exact nested equality covered:

- every model parameter;
- BN `running_mean`, `running_var` and `num_batches_tracked`;
- optimizer state and all momentum buffers;
- optimizer metadata;
- cursor `TrainingCursor(1, 5, 5)`;
- global update and LR;
- canonical final state fingerprint.

No tolerance relaxation was needed.

## H. Fresh-process replay

Two separately launched Python processes used seed 5 and five CUDA-generated
synthetic updates. Their complete one-line JSON outputs compared equal. Both
reported:

- fingerprint:
  `5826d596e2b1785aafec198161b7eea1da23663ec96ef8e1b633a4e3171301ae`;
- cursor: epoch 1, next batch 5, global update 5;
- exact RTX 3070 Ti environment identity;
- `cifar_optimizer_steps = 0`.

This establishes process-level replay for this machine/runtime/device combination.

## I. Synthetic batch-128 capacity

A synthetic float32 CUDA batch with shape `128 x 3 x 32 x 32` completed forward,
mean criterion and backward. Logits had shape `128 x 10`, the loss and logits were
finite, and full gradient coverage passed.

After resetting CUDA peak-memory statistics, the target reported:

| Metric | Bytes | Approximate GiB |
|---|---:|---:|
| peak allocated | `1,709,296,640` | 1.59 |
| peak reserved | `2,476,736,512` | 2.31 |

These measurements are preflight evidence only and were not used for tuning.

## J. CIFAR forward-only and Windows worker replay

The Phase 2 Windows worker regression compared `num_workers=0` with
`num_workers=2` and required identical sample order, labels and transformed
tensors. It passed in the full target suite.

One real Phase 2 training batch was copied CPU to CUDA and back with exact tensor
equality. The target batch evidence was:

| Field | Value |
|---|---|
| input shape | `[128, 3, 32, 32]` |
| input dtype | `torch.float32` |
| label dtype | `torch.int64` |
| output shape | `[128, 10]` |
| CIFAR backward count | `0` |
| CIFAR optimizer-step count | `0` |

The CIFAR path uses `model.eval()` and `torch.no_grad()` for model and criterion
forward only. It did not calculate accuracy or test error. An AST boundary test
independently rejects backward, step or `train_step` calls in the CIFAR-specific
Phase 4 production/test paths.

## K. Target test results

| Validation | Actual target result |
|---|---|
| Phase 4 boundary + CUDA tests | PASS - `12 passed, 0 skipped, 1 warning in 21.60s` |
| independent fresh-process pair | PASS - complete outputs identical |
| full repository pytest, first run | PASS - `140 passed, 0 skipped, 5 warnings in 48.83s` |
| full repository pytest, repeat | PASS - `140 passed, 0 skipped, 5 warnings in 49.55s` |
| Phase 4 CUDA tests executed | YES |
| Phase 4 CUDA tests skipped | `0` |

The warnings are the same torchvision CIFAR pickle-load
`VisibleDeprecationWarning` under NumPy 2.4. They did not change data assertions or
test outcomes.

## L. Frozen integrity and repository audit

The post-validation manifest audit matched every frozen binding:

| Artifact | Actual SHA-256 | Result |
|---|---|---|
| `references/wide_residual_networks.pdf` | `AF606AEB02AE3D713A3BA7AD34A5A583D911A5FD33E52B76507DC37A8ED021E9` | PASS |
| `AGENTS.md` | `2E169C8EA5B65685B5A8254AF6F9DF0C4F407062ECE47893C0BA240BF7D7471F` | PASS |
| `docs/reproduction_spec.md` | `7D471C9F90B6A172D25ABF83E8FBADD7CEC83656EA88EA4058DC3C0A22DC965F` | PASS |
| `docs/assumptions.md` | `A09A2D497FD0D72A1B14751124D2E9BD00C1F84A617E0CF38F0B56D7213F9794` | PASS |
| `docs/evidence_table.md` | `99E2B7D82BCA17CCDACC8E1E0D03FE90AB5EB160B80CFCA4A4F4E51C05780FA2` | PASS |
| `docs/open_questions.md` | `541C07906823843868E76E9458E53DAA89EC5F1AF048BE04D4D31B05E74CD987` | PASS |
| `configs/wrn16_8_arxiv_v4_frozen.yaml` | `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3` | PASS |

`compileall` and `git diff --check` passed after the Phase 4 document was added.
The Phase 3 artifact-boundary test remained part of the passing full suite. No
repository runtime path contains a `.pt`, `.pth` or `.ckpt` artifact, and no
`runs/`, `checkpoints/` or formal result output was created.

The static occurrence audit classified all matches as one of:

- synthetic-only Phase 1/3/4 backward or optimizer tests;
- the generic Phase 3 training-engine implementation;
- Phase 2 CIFAR dataset/transform construction;
- deterministic/TF32 policy checks, including the legacy API fallback;
- boundary-test prohibited-string sentinels.

No CIFAR-specific Phase 4 path contains backward or optimizer-step execution. No
AMP, GradScaler, compiled model or formal-training launcher was introduced.

## M. Phase boundary and authorization state

Phase 4 gates are `PASS` for the actual RTX 3070 Ti target. The repository is:

`READY FOR HUMAN FORMAL-TRAINING AUTHORIZATION`

This status is not authorization itself. At Phase 4 completion:

- formal CIFAR-10 training executed: **NO**;
- CIFAR backward count: **0**;
- CIFAR optimizer-step count: **0**;
- formal result generation executed: **NO**;
- hyperparameters changed: **NO**;
- Phase 5 started: **NO**.

Work stops after the independent Phase 4 commit and waits for human review.
