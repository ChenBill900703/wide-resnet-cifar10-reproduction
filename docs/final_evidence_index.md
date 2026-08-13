# WRN CIFAR-10 Final Evidence Index

Generated: 2026-08-13 (Asia/Taipei)
Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
Formal baseline: `225cf8d44c36a8a210f6989bf76b9ebfe460adbd`
Final evidence audit: **PASS**

SHA-256 values below were recomputed during finalization. `RECORDED + VERIFIED` means a file is covered by its run-local `artifact_hashes.json` and passed the independent 45/45 audit. This index does not include its own hash because embedding a file's hash inside itself is self-referential.

## 1. Primary paper and official-code evidence

| Path / locator | Purpose | SHA-256 / version | Status |
|---|---|---|---|
| `references/wide_residual_networks.pdf` | Local arXiv:1605.07146v4 primary paper; Table 5 target and 4.27% reference | `AF606AEB02AE3D713A3BA7AD34A5A583D911A5FD33E52B76507DC37A8ED021E9` | VERIFIED |
| `docs/evidence_table.md` | Claim-by-claim paper/code/dependency locators and conflict tracking | `99E2B7D82BCA17CCDACC8E1E0D03FE90AB5EB160B80CFCA4A4F4E51C05780FA2` | FROZEN / VERIFIED |
| `https://github.com/szagoruyko/wide-residual-networks/tree/ae6d0d0561484172790c7a63c8ce6ade5a5a2914` | Author official WRN repository pinned for Torch7 and later-official PyTorch evidence | commit `ae6d0d0561484172790c7a63c8ce6ade5a5a2914` | PINNED LOCATOR |
| `docs/evidence_table.md`, historical-dependency rows | Date-compatible Torch7 `nn`, `optim`, `cudnn.torch`, Torchnet and Torch RNG evidence; preserves actual-revision UNKNOWN status | source commits/lines recorded per row | VERIFIED LOCATORS |

## 2. Freeze and governance files

| Path | Purpose | SHA-256 | Status |
|---|---|---|---|
| `configs/wrn16_8_arxiv_v4_frozen.yaml` | Formal target and executable protocol | `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3` | FROZEN / VERIFIED |
| `docs/reproduction_spec.md` | Human-readable frozen specification | `7D471C9F90B6A172D25ABF83E8FBADD7CEC83656EA88EA4058DC3C0A22DC965F` | FROZEN / VERIFIED |
| `docs/assumptions.md` | Approved implementation assumptions and impact | `A09A2D497FD0D72A1B14751124D2E9BD00C1F84A617E0CF38F0B56D7213F9794` | FROZEN / VERIFIED |
| `docs/open_questions.md` | Remaining UNKNOWN items and non-claims | `541C07906823843868E76E9458E53DAA89EC5F1AF048BE04D4D31B05E74CD987` | FROZEN / VERIFIED |
| `docs/freeze_manifest.md` | Phase 0 paper/code/file binding | `217A4C0DDF6F423F1E8FFEEDAE254F2ABE3EBCF09A65996773D29F8E6E67075B` | VERIFIED |
| `data/cifar-10-python.tar.gz` | Frozen CIFAR-10 Python archive | `6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE` | VERIFIED |

## 3. Phase history and validation evidence

| Phase | Commit | Validation path | Validation SHA-256 | Status |
|---|---|---|---|---|
| Phase 0 | `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8` | `docs/freeze_manifest.md` | `217A4C0DDF6F423F1E8FFEEDAE254F2ABE3EBCF09A65996773D29F8E6E67075B` | SPEC FROZEN |
| Phase 1 | `99753b8cf2a5457faed8ddb695793a7f38e99729` | `docs/phase1_architecture_validation.md` | `440EF3F9DB047609DBE27100BA894C5FDE5C28A20376A1618AA6ED009519A86B` | PASS |
| Phase 2 | `f1ff88b532f4d77803b36e4f48fa1b407e01a1b7` | `docs/phase2_data_validation.md` | `C0A6E4C0B23A0ACFA54BB8F1916BE105312B6754A0C76F22CAA2FC7C1C19DC9A` | PASS |
| Phase 3 | `803f09a7a410318a8aea79a428746d63fdc817a7` | `docs/phase3_training_engine_validation.md` | `9EBEFF8271A03D9D299E050C3EACB377C8001A1A6FA4D3D4DF58F48E18535DB3` | PASS |
| Maintenance | `ee18d4c2ae54b41fb6f800a42c90c072f754355e` | Phase 3 boundary maintenance | commit identity | PASS |
| Maintenance | `c4dd870b328bf8bc3c5d5aed35910935de227447` | Local-environment boundary maintenance | commit identity | PASS |
| Phase 4 | `ef6363e0228fac51f70c77a79ab548e6fc60dc1f` | `docs/phase4_target_hardware_validation.md` | `85B995E4F487A39147CF00272708862FF0A99200083429633D961040B6D74C25` | PASS |
| Phase 5 | `225cf8d44c36a8a210f6989bf76b9ebfe460adbd` | `docs/phase5_formal_run_freeze.md` | `F261053E2A9D922471D9371C6AA692D5455A7DDBFC594E4ACC6A1F63EFE575F2` | FORMAL BASELINE FROZEN |
| Phase 5 environment | same baseline | `docs/phase5_environment_lock.txt` | `F2DFFC85B9DCD3D7F83266B1220D10C22A31F4E0AED866E9667ED95ED8407C42` | VERIFIED |

Historical validation counts recorded in those phase documents: Phase 1 `21 passed`; Phase 2 full `53 passed`; Phase 3 focused/full `74/127 passed`; Phase 4 targeted/full-twice `12/140 passed`; Phase 5 targeted/full `54/183 passed`. These are phase-time records, not substituted for the final artifact audit.

## 4. Formal run artifact roots

Formal root:

`D:\wide-resnet-formal-runs\WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`

Each exact seed directory contains `run_manifest.json`, `environment.json`, `training_log.jsonl`, `run_status.json`, `final_result.json`, `artifact_hashes.json`, and four checkpoints at epochs 60/120/160/200.

| Path | Purpose | SHA-256 handling | Status |
|---|---|---|---|
| `...\seed_01` | Formal Seed 1 complete artifact set | individual files bound by `seed_01/artifact_hashes.json` | VALID; 200 epochs; 78,000 updates; 0 resume |
| `...\seed_02` | Formal Seed 2 complete artifact set; median run | individual files bound by `seed_02/artifact_hashes.json` | VALID; 200 epochs; 78,000 updates; 0 resume |
| `...\seed_03` | Formal Seed 3 complete artifact set | individual files bound by `seed_03/artifact_hashes.json` | VALID; 200 epochs; 78,000 updates; 0 resume |
| `...\seed_04` | Formal Seed 4 complete artifact set | individual files bound by `seed_04/artifact_hashes.json` | VALID; 200 epochs; 78,000 updates; 0 resume |
| `...\seed_05` | Formal Seed 5 complete artifact set | individual files bound by `seed_05/artifact_hashes.json` | VALID; 200 epochs; 78,000 updates; 0 resume |
| `...\seed_01`…`seed_05\environment.json` | Exact hardware/software/numerical policy snapshots | RECORDED + VERIFIED | 5/5 PASS |
| `...\seed_01`…`seed_05\training_log.jsonl` | 207 events per run: epoch/LR/evaluation/final events | RECORDED + VERIFIED | 1,035/1,035 EVENTS PASS |
| `...\seed_01`…`seed_05\run_status.json` | Completion and resume status | RECORDED + VERIFIED | 5/5 COMPLETE; resume_count 0 |

## 5. Run manifests, hash manifests, and final results

| Seed | Exact path | Purpose | SHA-256 | Status |
|---|---|---|---|---|
| 1 | `...\seed_01\run_manifest.json` | Frozen target/baseline/config/environment/run plan | `B9596161FF046AD48F3537D144C85A55A697C72249DA4C41046E53ADD56BEA27` | VERIFIED |
| 1 | `...\seed_01\artifact_hashes.json` | Seed 1 artifact integrity manifest | `E0A71619EBFB49F49AB303A43C56124EEFE271CEDF6D4512D7B748032B8BA672` | VERIFIED |
| 1 | `...\seed_01\final_result.json` | 9,559 correct; 441 incorrect; error 4.41% | `9CDABE01319CBC615992B3746ECB654C3813548DDA872FA32E599473BF92EE2C` | VALID |
| 2 | `...\seed_02\run_manifest.json` | Frozen target/baseline/config/environment/run plan | `789EB253DDAD6607B65A206654A7E9964EE5B8DAEB12D10319E735527EF991B6` | VERIFIED |
| 2 | `...\seed_02\artifact_hashes.json` | Seed 2 artifact integrity manifest | `369C5F240FA1FA6CF26D0AC690ED534A46BD439CF211C0C02CC01F38F51EA30E` | VERIFIED |
| 2 | `...\seed_02\final_result.json` | 9,560 correct; 440 incorrect; error 4.40%; median run | `810E4DCFB1AF58B191DDF5CAF667C1EF8ADEAFC0097DEABA0790768BFBAFC4C1` | VALID / MEDIAN |
| 3 | `...\seed_03\run_manifest.json` | Frozen target/baseline/config/environment/run plan | `3D48E179F50030D5DEF2669D245AB3C90153C8936E6EB9E5FD12D05C36FC9F03` | VERIFIED |
| 3 | `...\seed_03\artifact_hashes.json` | Seed 3 artifact integrity manifest | `DAED1DDD056C53FED300551BEA20CC1BA85A67C85F797B34CB03EF23FEFE9C54` | VERIFIED |
| 3 | `...\seed_03\final_result.json` | 9,568 correct; 432 incorrect; error 4.32% | `E7E904661844596903680C7D4344C8D1326AF505F3E54297BF3D6B8D8E4E47B5` | VALID |
| 4 | `...\seed_04\run_manifest.json` | Frozen target/baseline/config/environment/run plan | `E5D5B1E06A1A73CBB9561CDCAE9F5AA954958EA9B346946222650870EC858B28` | VERIFIED |
| 4 | `...\seed_04\artifact_hashes.json` | Seed 4 artifact integrity manifest | `1DCD2BB5991D233F5AD64FDF6A9CCDDE6488499635AF79A3E20262DA3004E02D` | VERIFIED |
| 4 | `...\seed_04\final_result.json` | 9,572 correct; 428 incorrect; error 4.28% | `29855DD16B5DCDACF5F405484283B8CEA6488E9F09499A3CDB9AED55EA20BED2` | VALID |
| 5 | `...\seed_05\run_manifest.json` | Frozen target/baseline/config/environment/run plan | `33B24DCDDBE1EF81B4A8FBB2B2CF99D445FC3E7E5B64BFC0E0F8A06F1BF89852` | VERIFIED |
| 5 | `...\seed_05\artifact_hashes.json` | Seed 5 artifact integrity manifest | `246CFEA6E33C2DC787910B0A8BFB8703E9A434050BF67AAAECA6E454E6739D68` | VERIFIED |
| 5 | `...\seed_05\final_result.json` | 9,557 correct; 443 incorrect; error 4.43% | `8F7CE2FCF1425EA54E220BFACB42F6287A5EBCBB1E65E68C171B3EB010C2C3A1` | VALID |

## 6. Checkpoint artifact index

All checkpoints passed schema, tensor-finiteness, model/BN/optimizer/momentum, CPU/CUDA RNG, cursor, LR and metadata validation.

| Seed | Checkpoint path | SHA-256 | Status |
|---|---|---|---|
| 1 | `...\seed_01\checkpoints\epoch_060.pt` | `BE9A0BBA4CEC0D5B1F73F6D1A4AD0B7AE9F225439B53BAB34883E65CA85EBAC9` | VALID |
| 1 | `...\seed_01\checkpoints\epoch_120.pt` | `A2A75D659B202D61C75F15AA036F92C968B1CC357EB5AAC580637FB08740FCA9` | VALID |
| 1 | `...\seed_01\checkpoints\epoch_160.pt` | `4466CFEE9576360AA3919657621BC16D6C7C2260DA203C9882CB491D0D02B89C` | VALID |
| 1 | `...\seed_01\checkpoints\epoch_200.pt` | `4AB151DEBC161CD719F0E33B7B3676246B52CC1F8BAE68DA01E62DC030601F47` | VALID / FINAL |
| 2 | `...\seed_02\checkpoints\epoch_060.pt` | `F89191F654EA37A6DD51D6DFD0E5F87558CDAE4ED624E8CD4A9FA27DF296C476` | VALID |
| 2 | `...\seed_02\checkpoints\epoch_120.pt` | `2ED7053706979B921F783B3D385BF934A4E8A28BB0188549F4D07B983EBCCDD5` | VALID |
| 2 | `...\seed_02\checkpoints\epoch_160.pt` | `620C27924031577B26AB4FF832DECB405D88BD26C7507F02B4AE3DB356FB7964` | VALID |
| 2 | `...\seed_02\checkpoints\epoch_200.pt` | `B26BB0E52E2B10D808BE7B8BE724624D1DA0338A84BC9DD181FF8559FC2217FD` | VALID / FINAL / MEDIAN RUN |
| 3 | `...\seed_03\checkpoints\epoch_060.pt` | `299FDA276340954671E238D91EC0CE1743D589A7AF3D64367CD6537F767C3D63` | VALID |
| 3 | `...\seed_03\checkpoints\epoch_120.pt` | `0F7FAF0C679A3E735AD1AAA80C14124258EECC03F332CCE965F62CFCB057CF62` | VALID |
| 3 | `...\seed_03\checkpoints\epoch_160.pt` | `4862D2951F0B70F2065BC08CFCBE0871EA31D12E45522A170EBA7F2C674414A9` | VALID |
| 3 | `...\seed_03\checkpoints\epoch_200.pt` | `7D60439E946680B8EE794157817581BFB560E98075FC754BF6EACAA173984DEA` | VALID / FINAL |
| 4 | `...\seed_04\checkpoints\epoch_060.pt` | `B18D2F54FBE1DEBFB224F6C7BD04D76F81383318654004AA5435612F61191C69` | VALID |
| 4 | `...\seed_04\checkpoints\epoch_120.pt` | `FB667B208D23E6796F43D23AC586045F1B87E820A252B6CF76A2C864FDC6E8C7` | VALID |
| 4 | `...\seed_04\checkpoints\epoch_160.pt` | `CA61096B507D7A49840F7C5055AFAD7B4CB894C5D8D962565DC064E4B25D08E3` | VALID |
| 4 | `...\seed_04\checkpoints\epoch_200.pt` | `6BB022C606978E106337AF217B8EE7E512E62F5FD0578A52F9A40716A1D464F2` | VALID / FINAL |
| 5 | `...\seed_05\checkpoints\epoch_060.pt` | `BCC227D1E089570C1FEEA219B85EF004BD895D8F24A3BAD111A0AE2CA897EB93` | VALID |
| 5 | `...\seed_05\checkpoints\epoch_120.pt` | `B6A2E7011C38EC026ADD55397D5CF10B296DC0B92D85C75E9F5CA813333B1ABE` | VALID |
| 5 | `...\seed_05\checkpoints\epoch_160.pt` | `2D747283C5A86EF7D5078378F7E29C94F54087D871488BF65F3E9A66BA3ED26B` | VALID |
| 5 | `...\seed_05\checkpoints\epoch_200.pt` | `D0144AF2F2119DA2E2D8B4118A91AD1E9C6E47948CD5AA213B98BF5C67506FF1` | VALID / FINAL |

Checkpoint total: **20/20 VALID**.

## 7. Median calculation evidence

| Path / value | Purpose | SHA-256 | Status |
|---|---|---|---|
| `docs/formal_median_transcript.json` | Human-readable/machine-readable input mapping, sorted errors, median and comparison | file SHA `3BE62CC5C06A479D6D9C6BE1C405EFA33ED88C90A4049DD6C5E0A0AB877E2C15` | GENERATED / VERIFIED |
| Canonical median payload | Exact canonical calculation payload used for transcript verification | `993C9057DFF5B265DCE0E6087AB63628CC1DE2E272D0D2DC7D263D0DCD28C49D` | VERIFIED |
| Sorted formal errors | `4.28, 4.32, 4.40, 4.41, 4.43` | derived from five exact final results | VERIFIED |
| Frozen median | `4.40%`, Seed 2, accuracy 95.60%, 9,560/440 | pre-registered median rule | VERIFIED |
| Paper comparison | 4.40% − 4.27% = `+0.13 pp` | derived numerical comparison | VERIFIED; no tolerance classification |

The transcript **file SHA** and **canonical payload SHA** are intentionally different concepts. The former hashes the complete JSON file; the latter hashes the canonical calculation payload recorded inside it.

## 8. Final documentation and presentation package

| Path | Purpose | SHA-256 | Status |
|---|---|---|---|
| `docs/final_reproduction_report.md` | 18-section final reproduction report plus 30-second and 2-minute summaries | `775C7F9223FE97089009489C0B69532B7E4452AF0A6DB6D59466BADF92FD1075` | FINAL |
| `docs/final_claim_matrix.md` | Evidence classification and safe/unsafe wording guard | `40C908F58560BB4162F73C711CB0304FA762F39F2546FF684CCD70BA90D107D2` | FINAL |
| `docs/professor_defense_qa.md` | 45 grouped questions + 12 adversarial questions | `D5304BF9F3A3BCD7BFF7D6EF913ADABF27362D957F5D34BE9FEA8879C1DF5BD7` | FINAL |
| `docs/wrn_presentation_script.md` | Complete Traditional Chinese script for all 16 slides; target 14:10 | `96037D6E39A31820AB0ECB05BB76E8FEAA51415072A08F773FCC79A6E91F9786` | FINAL |
| `docs/final_project_handoff.md` | Delivery state, operating notes, artifact preservation and stop condition | `69B71CF22AAF450967672EBFFD2C850C824822F3AD12434BF1D16F2C98831B78` | FINAL |
| `WRN_CIFAR10_Reproduction_Final.pptx` | Editable 16:9, 16-slide professor deck with native shapes/tables/charts and speaker notes | `AB2E04D7B49F86669DE25073039D528696E58F2E6F1B1B6C945DB1378A1C9667` | FINAL; VISUAL QA PASS; NO OVERFLOW |
| `docs/final_evidence_index.md` | This complete delivery and artifact map | self-hash intentionally omitted | FINAL |

## 9. Final audit statement

| Check | Result |
|---|---|
| Git formal baseline | `225cf8d44c36a8a210f6989bf76b9ebfe460adbd` VERIFIED |
| Frozen YAML | `18EF6815...959B3` VERIFIED |
| Dataset archive | `6D958BE0...1CE` VERIFIED |
| Paper PDF | `AF606AEB...21E9` VERIFIED |
| Formal runs | 5/5 VALID |
| Formal artifact hashes | 45/45 PASS |
| Checkpoints | 20/20 PASS |
| Log events | 1,035/1,035 PASS |
| Epoch/update/resume | each 200 / 78,000 / 0 VERIFIED |
| Median | 4.40% recomputed |
| Canonical median transcript SHA | `993C9057...C49D` VERIFIED |
| PPTX slide count | 16 |
| PPTX visual inspection | all 16 slides individually inspected |
| PPTX overflow test | PASS; no overflow detected |
| Repository regression suite | `183 passed`, 5 known torchvision/NumPy warnings |
| `python -m compileall -q src tests` | PASS |
| `git diff --check` | PASS |
| Post-hoc performance tolerance | NONE introduced |
| Formal training during finalization | NONE |
| Formal artifact modification during finalization | NONE |

Final professor-readiness: **READY**.
