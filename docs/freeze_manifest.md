# Phase 0 freeze manifest

- Frozen target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Freeze version: `v1`
- Freeze date/time: `2026-08-10T14:06:20.6497894+08:00` (Asia/Taipei)
- Pre-freeze Git HEAD: `UNBORN` (the repository was initialized for this freeze and had no prior commit)
- Official source commit: `ae6d0d0561484172790c7a63c8ce6ade5a5a2914`

## Frozen artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `references/wide_residual_networks.pdf` | `AF606AEB02AE3D713A3BA7AD34A5A583D911A5FD33E52B76507DC37A8ED021E9` |
| `AGENTS.md` | `2E169C8EA5B65685B5A8254AF6F9DF0C4F407062ECE47893C0BA240BF7D7471F` |
| `docs/reproduction_spec.md` | `7D471C9F90B6A172D25ABF83E8FBADD7CEC83656EA88EA4058DC3C0A22DC965F` |
| `docs/assumptions.md` | `A09A2D497FD0D72A1B14751124D2E9BD00C1F84A617E0CF38F0B56D7213F9794` |
| `docs/evidence_table.md` | `99E2B7D82BCA17CCDACC8E1E0D03FE90AB5EB160B80CFCA4A4F4E51C05780FA2` |
| `docs/open_questions.md` | `541C07906823843868E76E9458E53DAA89EC5F1AF048BE04D4D31B05E74CD987` |
| `configs/wrn16_8_arxiv_v4_frozen.yaml` | `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3` |

## Integrity and interpretation

- Hashes were computed after YAML parsing, frozen-value consistency checks, and Markdown table-column checks passed.
- The local PDF is evidence but is intentionally not included in the Phase 0 Git commit; its content hash binds the manifest to the reviewed artifact.
- Human approval selects project policy but does not upgrade implementation assumptions to paper or official-code facts.
- Any change to a hashed artifact invalidates this manifest and requires a new freeze version and explicit human approval.
