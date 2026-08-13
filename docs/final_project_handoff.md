# WRN CIFAR-10 Final Project Handoff

## Delivery state

- Final evidence audit: **PASS**
- Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Formal code baseline: `225cf8d44c36a8a210f6989bf76b9ebfe460adbd`
- Frozen config SHA-256: `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3`
- Dataset archive SHA-256: `6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE`
- Formal runs: **5/5 VALID**
- Artifact validation: **45/45 hashes**, **20/20 checkpoints**, **1,035/1,035 log events**
- Every run: **200 epochs**, **78,000 optimizer updates**, **0 resume**
- Frozen aggregation: five-run median of epoch-200 final-checkpoint CIFAR-10 test errors
- Frozen median: **4.40%** (Seed 2; 9,560 correct / 440 incorrect)
- Paper reference: **4.27%**
- Transparent numerical difference: **+0.13 percentage points**

## Required conclusion wording

> Protocol-faithful reproduction with pre-registered modern-framework assumptions, numerically compared against paper 4.27% median. Observed frozen five-run median test error: 4.40%.

沒有預先核准的 performance tolerance、equivalence margin 或 significance test。因此不得事後宣稱：

- 「完全重現論文 4.27%」；
- performance pass / failure；
- statistical equivalence / insignificance；
- 某一 modern-framework 差異已被證明造成 +0.13 pp。

## Primary handoff artifacts

| Artifact | Purpose | State |
|---|---|---|
| `docs/final_reproduction_report.md` | 完整研究、技術、結果與限制報告 | Final |
| `docs/final_claim_matrix.md` | 答辯可說／不可說的 evidence guard | Final |
| `docs/professor_defense_qa.md` | 45 題主題題＋12 題 adversarial defense bank | Final |
| `docs/wrn_presentation_script.md` | 16 頁完整口說稿，預計 14:10 | Final |
| `docs/formal_median_transcript.json` | 可機器重算的 frozen median transcript | Final |
| `docs/final_evidence_index.md` | source、freeze、formal artifacts 與 delivery hashes | Final |
| `WRN_CIFAR10_Reproduction_Final.pptx` | 16:9、16 頁、可編輯、含 speaker notes 的教授簡報 | Final |

## Presentation operating notes

1. 以 Slide 1 的 4.40% / 4.27% 定義研究範圍。
2. Slide 6 是 evidence classification 防守核心；被問到「這數字從哪裡來」時先定位 classification。
3. Slide 12 必須先報五個 individual runs，再報 median；不得挑 Seed 4 的 4.28%。
4. Slide 13 使用從 0 起算的公平 y-axis；說「+0.13 percentage points」，不是「相對差 0.13%」。
5. 被問「為何不調到 4.27%」時，回答 freeze-before-results 與 no post-hoc tuning。
6. 不確定時使用 `docs/final_claim_matrix.md` 的 safe wording；來源沒有支持就標 `UNKNOWN`。

## Artifact preservation

正式訓練產物位於 repository 外的：

`D:\wide-resnet-formal-runs\WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`

本次 finalization 只讀取這些 formal artifacts；沒有重新訓練，也沒有修改正式 results、logs、manifests 或 checkpoints。若要搬移或封存，應保留完整 seed directories 與其 `artifact_hashes.json`，搬移後重新做 SHA-256 audit。

## Maintenance boundary

- Frozen specification/config 或正式 baseline 若要改動，必須另開版本與新的人工 freeze；不得覆寫 v1。
- 新增 seeds、改 aggregation、改 checkpoint selection 或改 tolerance 都是新研究，不得回寫本次 4.40% formal result。
- PowerPoint 是可編輯交付檔；外部逐頁講稿 `docs/wrn_presentation_script.md` 為口說內容的權威版本。
- 下一個 DenseNet project 應重新建立 evidence table、source lineage、assumption approval 與 freeze；不得直接沿用 WRN 的未知項或結論。

## Final stop condition

本專案已完成最終文件與簡報封裝。**不要開始新的 training run；等待人工 review。**
