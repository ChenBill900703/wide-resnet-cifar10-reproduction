# Phase 2 CIFAR-10 data-pipeline validation

Status: `PASS — waiting for human Phase 2 review; no Phase 3 authorization`

- Frozen target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Human Phase 1 approval: 2026-08-10; Phase 1 is closed
- Phase 0 freeze commit: `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8`
- Phase 1 commit: `99753b8cf2a5457faed8ddb695793a7f38e99729`
- Phase 2 commit: reported in the final handoff because a commit cannot embed its own final SHA
- Frozen config SHA-256: `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`

Phase 2 is limited to dataset acquisition, preprocessing, augmentation, deterministic
data RNG, sampling, loading and their validation. It does not authorize optimization,
training, checkpoints, result evaluation or Phase 3.

## Runtime and installed data implementation

- Python: `3.12.13`
- PyTorch: `2.13.0+cpu`
- torchvision: `0.28.0+cpu` (distribution metadata version `0.28.0`)
- Runtime: Windows 11 CPU workspace runtime
- Target RTX 3070 Ti hardware: `NOT VERIFIED IN TARGET HARDWARE`
- Dataset implementation: installed `torchvision.datasets.CIFAR10`
- Source locator: installed module `torchvision/datasets/cifar.py`, class `CIFAR10`
- Source access/audit date: 2026-08-10

torchvision `0.28.0` is the release paired with the existing PyTorch `2.13.0`
runtime. It was installed at the exact version without replacing the existing torch
package.

## Dataset source and integrity contract

- Download URL from the installed implementation:
  `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
- Archive filename: `cifar-10-python.tar.gz`
- torchvision archive MD5: `c58f30108f718f92721af3b95e74349a`
- Extracted folder: `cifar-10-batches-py`
- Local root: repository-local ignored `data/`
- Project-level archive SHA-256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`

Installed torchvision integrity members:

| Split | File | torchvision MD5 |
|---|---|---|
| train | `data_batch_1` | `c99cafc152244af753f735de768cd75f` |
| train | `data_batch_2` | `d4bba439e000b95fd0a9bffe97cbabec` |
| train | `data_batch_3` | `54ebc095f3ab1f0389bbae665268c751` |
| train | `data_batch_4` | `634d18415352ddfa80567beed471001a` |
| train | `data_batch_5` | `482c414d41f54cd18b22e5b47cb7c3cb` |
| test | `test_batch` | `40351d587109b95175f43aff81a1287e` |
| metadata | `batches.meta` | `5ff9c542aee3614f3951f8cda6e48888` |

These MD5 values are installed torchvision implementation constants, not paper
checksums. The independently calculated SHA-256 is a project artifact identity, not a
paper-provided value.

Expected and validation-gated dataset identity:

- Train examples: 50,000 from the five train batch files
- Test examples: 10,000 from `test_batch`
- Raw storage: uint8 RGB, HWC `(N,32,32,3)` in torchvision
- Wrapper output: float32 RGB, CHW `(3,32,32)`
- Labels: Python integer class indices 0 through 9
- Class metadata SHA-256 over compact JSON list:
  `feb528d8ad763e4c89a96910f8ed9ea0113aa59fe89528e3f8804b4ae9b0d2e5`
- Actual artifact/result: `PASS`; the archive MD5, all six split-file MD5s and
  metadata MD5 match the installed torchvision constants. The loaded train/test
  arrays contain 50,000/10,000 examples with the expected shape, dtype, labels and
  class metadata hash.

No dataset statistics are fitted or recomputed. The test set is not used to select,
adapt or tune any transform.

## Frozen normalization mapping

Raw torchvision uint8 HWC images are copied to RGB CHW tensors, converted to
`torch.float32` without division, and normalized on the frozen 0..255 scale:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| R | 125.3 | 63.0 |
| G | 123.0 | 62.1 |
| B | 113.9 | 66.7 |

The implemented expression is channel-wise `(x - mean) / std`. Tests lock mean to
zero, mean plus/minus one standard deviation to plus/minus one, RGB/CHW layout,
float32 output and an explicit failure detector for applying byte-scale statistics
directly to 0..1 input.

## Frozen train augmentation mapping

Actual transform order:

1. Horizontal flip decision.
2. Four-sided edge-excluding reflection padding by 4.
3. Random 32x32 crop from the padded 40x40 tensor.
4. Float32 conversion preserving 0..255 values.
5. Frozen channel-wise normalization.

The flip draw is an explicit uniform integer in `{0,1}`: draw 0 flips and draw 1
leaves the image unchanged. Width and height crop offsets are separate uniform integer
draws in `{0,...,8}`; offset 9 is rejected. A manually constructed tensor test proves
that `torch.nn.functional.pad(..., mode="reflect")` matches the adopted
edge-excluding historical boundary map. Controlled asymmetric-image tests distinguish
the frozen flip-before-pad/crop order from pad/crop-before-flip.

The test transform performs only float32 byte-scale normalization. It has no flip,
padding, crop, resize or stochastic state.

## Deterministic data RNG realization

The modern port deliberately does not claim the unknown Torch7 two-worker stream.
It implements the approved deterministic realization:

- zero-based epoch `e0` in 0..199 maps to data seed `e0+1` in 1..200;
- the epoch seed drives a local CPU `torch.Generator` for the shuffled sample order;
- SplitMix64 with fixed integer constants derives a per-sample seed from the epoch
  data seed and numeric sample index;
- a separate local generator produces the sample's flip, x-offset and y-offset;
- Python `hash()`, global RNG state and worker scheduling are not inputs;
- `set_epoch(e0)` updates both the dataset epoch and sampler epoch;
- the dataset epoch is stored in shared CPU memory for worker visibility.

Therefore the same epoch and sample identity replay the same augmentation regardless
of unrelated item reads. Worker-count equivalence is validated between zero and two
workers on Windows. The worker count remains an implementation/runtime policy, not a
paper setting.

## DataLoader contract

- Train batch size: 128
- Train shuffle: explicit epoch-seeded sampler
- Train `drop_last=True`
- Train batches per epoch: `floor(50000/128)=390`
- Train samples presented per epoch: `390*128=49,920`
- Test order: deterministic source order
- Test `drop_last=False`
- Test batches: `ceil(10000/128)=79`, a derived runtime check
- Test examples covered: all 10,000 exactly once

The implementation contains no hidden dataset download default. Dataset acquisition
requires an explicit `download=True`; ordinary construction defaults to false.

## Validation results

| Validation | Result |
|---|---|
| Dataset archive/extracted-file integrity | PASS — archive MD5 and all extracted member MD5s match; archive SHA-256 recorded above |
| Dataset metadata and split sizes | PASS — 50,000 train, 10,000 test, 10 classes, uint8 RGB HWC source |
| Normalization tests | PASS |
| Manual reflection/crop/order probes | PASS |
| Epoch/sample replay | PASS |
| Actual full loader and test coverage | PASS — 390 train batches/49,920 examples; 79 test batches/all 10,000 examples |
| Zero-worker/two-worker equivalence | PASS — sample order, labels and transformed tensors are identical |
| Full repository pytest | PASS — 53 passed in 28.36s |
| Phase 1 regression suite | PASS — 21 passed in 3.49s |
| `compileall` | PASS |
| `git diff --check` | PASS |
| Frozen artifact hash recheck | PASS — frozen config SHA-256 equals the expected manifest value |

## Phase boundary

- Optimizer implementation: absent
- Scheduler implementation: absent
- Training loop or `optimizer.step()`: absent
- Model training/backward in Phase 2: absent
- Checkpoints or runs: absent
- Formal/smoke training: not performed
- Test accuracy or model-performance evaluation: not performed
- Phase 3: not started and not authorized
