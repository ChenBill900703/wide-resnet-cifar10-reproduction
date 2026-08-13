# WRN CIFAR-10 Final Claim Matrix

本矩陣規範最終報告、簡報與口試可使用的說法。`Safe wording` 可直接使用；`Unsafe wording` 代表超出證據支持範圍或混淆分類，禁止使用。

| Claim | Classification | Evidence source | Exact location / file | Safe wording | Unsafe wording |
|---|---|---|---|---|---|
| Formal target 為 WRN-16-8、CIFAR-10、mean/std、no dropout | `FROZEN PROJECT DECISION`，其組成有 paper/code 證據 | arXiv v4；frozen spec/config | paper p.8 Table 5；`docs/reproduction_spec.md` §1–2；`configs/wrn16_8_arxiv_v4_frozen.yaml` | 「本專案預先凍結的 formal target 是 WRN-16-8 mean/std no-dropout 家族。」 | 「論文只允許這一種 WRN-16-8 設定。」 |
| 論文參考 error 為 4.27% | `PAPER-SPECIFIED` | arXiv:1605.07146v4 | `references/wide_residual_networks.pdf`, p.8, Table 5, WRN-16-8 row | 「arXiv v4 Table 5 報告 4.27% test error。」 | 「任何 WRN-16-8 都應達到 4.27%。」 |
| 4.27% 是五次執行中位數 | `PAPER-SPECIFIED` | arXiv v4 | paper p.8, Table 5 caption: results are medians of five runs | 「表中結果是五次執行的中位數。」 | 「論文使用 seeds 1–5。」 |
| 論文參數量為 11.0M | `PAPER-SPECIFIED` | arXiv v4 | paper p.8, Table 5 | 「論文表格以一位小數報告 11.0M。」 | 「論文精確參數量就是 11,000,000。」 |
| 本實作精確參數量為 10,961,370 | `IMPLEMENTATION-VALIDATED`; `DERIVED` rounding cross-check | model parameter audit | `docs/phase1_architecture_validation.md`; Phase 1 tests | 「本實作精確計數 10,961,370，四捨五入為 11.0M，與論文粒度一致。」 | 「參數量相同證明整個訓練語義完全相同。」 |
| WRN 以增加寬度減少單純加深的效率問題 | `PAPER-SPECIFIED` conceptual claim | arXiv v4 | paper pp.1–3, abstract and §3 | 「論文主張 widening 可改善深度殘差網路的效率與表現權衡。」 | 「WRN 在所有任務都優於更深 ResNet。」 |
| CIFAR depth 公式為 `n = 6N + 4` | `OFFICIAL-CODE-SPECIFIED` and paper-supported architecture relation | paper + author Torch7 code evidence table | `docs/evidence_table.md`, architecture depth formula rows; paper §3 | 「WRN CIFAR family 使用 `n=6N+4`；WRN-16 對應每 stage 2 blocks。」 | 「這個公式適用所有 ResNet 變體。」 |
| Stage widths 為 16、16k、32k、64k | `PAPER-SPECIFIED` | arXiv v4 | paper p.4, Table 1 | 「論文 Table 1 定義各 stage channel widths。」 | 「這些 widths 是由 PyTorch 慣例決定。」 |
| WRN-16-8 的 stage widths 為 128、256、512 | `DERIVED FROM PAPER-SPECIFIED` | Table 1 widths and k=8 | `docs/reproduction_spec.md` architecture section | 「代入 k=8 可推得 128、256、512。」 | 「論文表格逐字列出 128、256、512。」 |
| Block family 為 `B(3,3)` | `PAPER-SPECIFIED` | arXiv v4 | paper p.4, Table 1；p.8, Table 5 context | 「正式 target 使用論文的 `B(3,3)` residual block family。」 | 「任何兩層 3×3 PyTorch block 都與 paper-era code 等價。」 |
| Shortcut 採 identity 或 projection routing | `OFFICIAL-CODE-SPECIFIED` | author implementation | `docs/evidence_table.md`, shortcut rows；`docs/reproduction_spec.md` | 「shortcut 行為依 frozen official-code evidence 實作並經單元測試。」 | 「論文正文唯一決定每個 shortcut 的所有細節。」 |
| Dropout 為 0 | `PAPER-SPECIFIED` target variant | arXiv v4 | paper p.8, Table 5 caption/row family；frozen config | 「選定的是 no-dropout target，dropout probability 凍結為 0。」 | 「WRN 論文不使用 dropout。」 |
| CIFAR-10 為 50,000 train / 10,000 test | `PAPER-SPECIFIED` | arXiv v4 | paper p.6, §4.2 | 「論文說明 CIFAR-10 的 train/test 規模。」 | 「這證明下載檔案未損壞。」 |
| Dataset archive integrity 通過 | `PROJECT-VALIDATED` | archive SHA check | formal manifests；`docs/phase5_formal_run_freeze.md` | 「本地 archive SHA-256 與 frozen 值相符。」 | 「SHA 相符證明資料標註本身完全正確。」 |
| Preprocessing 使用 per-channel mean/std | `PAPER-SPECIFIED` family + official-code micro-semantics | paper + author code | paper p.8 Table 5；`docs/evidence_table.md`, preprocessing rows | 「formal target 是 mean/std family；數值與運算順序依 frozen code evidence。」 | 「論文正文完整列出所有 normalization 常數與浮點細節。」 |
| Augmentation 使用 reflection-pad/crop 與 horizontal flip family | `PAPER-SPECIFIED` family + `OFFICIAL-CODE-SPECIFIED` micro-semantics | paper + author code | paper p.6 §4.2；`docs/reproduction_spec.md` augmentation section | 「augmentation family 有 paper 支持；邊界與 RNG 細節由 frozen official code/assumptions決定。」 | 「現代 torchvision 預設 augmentation 就等於 paper-era pipeline。」 |
| Batch size 為 128 | `PAPER-SPECIFIED` / frozen evidence | arXiv v4 training description | `docs/evidence_table.md`, batch-size row；frozen config | 「formal batch size 凍結為 128。」 | 「batch size 128 是為 RTX 3070 Ti 調出的最佳值。」 |
| 每 epoch 390 updates | `DERIVED` | 50,000 samples, batch 128, drop-last behavior | `docs/reproduction_spec.md`; formal metadata | 「依 frozen sampling/drop-last 規則，每 epoch 為 390 updates。」 | 「論文直接報告 390 updates。」 |
| 200 epochs 共 78,000 updates | `DERIVED` | 390 × 200 | frozen config；all run metadata | 「390×200 得 78,000 optimizer updates。」 | 「論文直接報告 PyTorch global update 78,000。」 |
| Optimizer 為 SGD、momentum 0.9、weight decay 5e-4 | Mixed: paper/code/dependency-backed, frozen | evidence table and frozen spec | `docs/evidence_table.md`, optimizer/momentum/weight-decay rows；frozen config | 「這組 optimizer 參數在 freeze 前逐項分類並固定。」 | 「所有細節都由單一論文段落完整指定。」 |
| LR 為 0.1，於 epochs 60/120/160 乘 0.2 | Evidence-backed and frozen | paper/official code audit | `docs/reproduction_spec.md` LR schedule；frozen config | 「formal schedule 是 0.1→0.02→0.004→0.0008，邊界預先凍結。」 | 「這是為了讓本次結果接近 4.27% 才選的。」 |
| Seeds 為 `[1,2,3,4,5]` | `IMPLEMENTATION-ASSUMPTION`, human-approved | assumptions/frozen config | `docs/assumptions.md`; `configs/wrn16_8_arxiv_v4_frozen.yaml` | 「論文未揭露 seed 值；本專案預先核准 seeds 1–5。」 | 「論文原始 runs 使用 seeds 1–5。」 |
| 只採 epoch-200 final checkpoint | `IMPLEMENTATION-ASSUMPTION`, human-approved | assumptions/frozen spec | `docs/assumptions.md`; `docs/reproduction_spec.md`; run manifests | 「checkpoint selection 在訓練前固定為 epoch 200 final only。」 | 「論文明說不可使用其他 checkpoint。」 |
| 未使用 tuning 或挑 seed | `PROJECT-PROCESS CLAIM`, audit-backed | freeze history and five-run manifests | Git commits through `225cf8d...`; five seed artifact sets | 「freeze 後以預登記 seeds 全數執行，沒有依結果調參或排除 seed。」 | 「沒有人曾看過任何中間結果。」 |
| Formal baseline commit 為 `225cf8d...` | `PROJECT-VALIDATED` | Git and run manifests | Git commit `225cf8d44c36a8a210f6989bf76b9ebfe460adbd`; every formal manifest | 「五個 runs 均記錄同一 frozen code SHA。」 | 「SHA 相同本身證明硬體執行逐 bit 相同。」 |
| Formal runtime 使用 RTX 3070 Ti | `ENVIRONMENT-OBSERVED` | environment snapshots | each seed `manifest.json`; Phase 4/5 validation docs | 「本次正式執行環境記錄為 RTX 3070 Ti、compute capability 8.6。」 | 「論文使用 RTX 3070 Ti。」 |
| Runtime 採 FP32，停用 AMP、TF32 與 compile | `PROJECT-FROZEN POLICY`, runtime-validated | numerical-policy snapshots | each manifest；`docs/phase4_target_hardware_validation.md`；`docs/phase5_formal_run_freeze.md` | 「formal runtime 明確鎖定 FP32 eager deterministic policy。」 | 「這與 2016 Torch7/CUDA runtime 在所有數值上等價。」 |
| 五個 seed errors 是 4.41、4.40、4.32、4.28、4.43% | `FORMAL OBSERVATION` | epoch-200 final results | external formal run `seed_01`…`seed_05` `final_result.json`; `docs/formal_median_transcript.json` | 「五個完整 formal runs 觀測到這五個 test errors。」 | 「這五個值是論文的 individual runs。」 |
| Frozen median 為 4.40% | `DERIVED FROM FORMAL OBSERVATIONS` | pre-registered median rule | `docs/formal_median_transcript.json`; sorted result values | 「排序後中央值為 4.40%，對應 Seed 2。」 | 「4.40% 是最佳 seed。」 |
| 與 paper reference 差 +0.13 percentage points | `DERIVED` | 4.40 − 4.27 | `docs/formal_median_transcript.json` | 「純數值差為 +0.13 percentage points。」 | 「相對誤差是 0.13%。」 |
| 可宣稱 protocol-faithful reproduction | `PROJECT CONCLUSION`, scoped | evidence audit + frozen process | `docs/final_reproduction_report.md`; phase validation docs | 「這是具預登記現代框架假設的 protocol-faithful reproduction。」 | 「這是 paper-era Torch7 run 的 bitwise exact replication。」 |
| 不能宣稱 performance pass/fail | `LIMITATION` | no preapproved tolerance | `docs/reproduction_spec.md`; `docs/open_questions.md`; final report | 「沒有事前 tolerance，因此只報數值差，不追認 pass/fail。」 | 「差 0.13 pp 所以已成功／失敗重現。」 |
| 不能宣稱統計等價或顯著差異 | `LIMITATION` | no inferential test/predefined equivalence margin | `docs/final_reproduction_report.md` limitations | 「五-run median 可比較，但不足以在未預註冊統計設計下宣稱等價或顯著。」 | 「0.13 pp 可忽略、統計上無差異。」 |
| 45/45 artifacts、20/20 checkpoints、1,035/1,035 log events 通過 | `PROJECT-VALIDATED` | independent final artifact audit | final audit output summarized in `docs/final_reproduction_report.md`; hash manifests | 「最終完整性稽核全部通過。」 | 「通過 hash audit 證明模型達到論文性能。」 |
| 五個 runs 均未 resume | `FORMAL ARTIFACT OBSERVATION` | manifests/log structure | five run manifests and logs；final audit | 「resume_count 均為 0，所有 runs 從初始狀態連續完成。」 | 「沒有任何作業系統或硬體中斷。」 |
| Checkpoints 含有限 model/optimizer state 與 RNG/cursor | `PROJECT-VALIDATED` | checkpoint semantic audit | 20 checkpoint files；final audit summary | 「20/20 checkpoints 的主要 tensors、BN、momentum、RNG 與 cursor metadata 均通過。」 | 「因此在任何軟硬體環境都保證 bitwise replay。」 |
| 論文未揭露原始五個 seeds | `UNKNOWN` | source audit | `docs/open_questions.md`; `docs/evidence_table.md` | 「paper/code evidence 未唯一提供原始 seed identities。」 | 「作者一定使用隨機 seeds。」 |
| Paper-era完整軟體鎖檔與 GPU 環境未知 | `UNKNOWN` | source audit | `docs/open_questions.md`; final report limitations | 「無法宣稱 runtime-identical；本專案完整記錄自己的現代環境。」 | 「現代 PyTorch 環境與作者環境相同。」 |
| 差異可能來自 framework/RNG/kernel/runtime semantics | `PLAUSIBLE CONTRIBUTORS`, not causal findings | documented limitations | `docs/final_reproduction_report.md` differences/limitations | 「這些是合理的可能來源，未做因果歸因。」 | 「0.13 pp 已證明是 CUDA 或 PyTorch 版本造成。」 |
| 下一步可用同一證據方法研究 DenseNet | `PROJECT LESSON / FUTURE WORK` | methodological inference | `docs/final_reproduction_report.md` lessons | 「可延用 source separation、freeze、multi-run audit 方法。」 | 「本 WRN 結果直接證明 DenseNet 也會重現成功。」 |

## Final wording lock

正式口頭與書面結論建議固定為：

> 本專案完成一個具預先登記現代框架假設、可稽核的 WRN-16-8 CIFAR-10 protocol-faithful reproduction。五個正式 runs 的 frozen median test error 為 4.40%，相較 arXiv v4 Table 5 的 4.27% 高 0.13 percentage points。因未預先核准 performance tolerance 或統計等價界線，本專案不事後宣告 performance pass、failure、equivalence 或 insignificance。
