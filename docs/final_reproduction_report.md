# Wide Residual Networks CIFAR-10 最終重現報告

Formal target: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`

Final evidence audit: **PASS**
Formal runs: **5/5 VALID**
Frozen five-run median test error: **4.40%**
Paper reference: **4.27%**
Numerical difference: **+0.13 percentage points**

> Protocol-faithful reproduction with pre-registered modern-framework assumptions, numerically compared against paper 4.27% median. Observed frozen five-run median test error: 4.40%.

## 1. Executive summary

本專案以 Zagoruyko 與 Komodakis 的 *Wide Residual Networks* 為研究對象，正式目標綁定 arXiv:1605.07146v4 第 8 頁 Table 5 的 CIFAR-10 WRN-16-8、mean/std normalization、no-dropout 結果家族。論文報告的參考值為五次執行中位數 test error **4.27%**，參數量為四捨五入後 **11.0M**。這兩個數值屬於 `PAPER-SPECIFIED`。

重現使用預先凍結的 PyTorch 現代化實作與 RTX 3070 Ti 環境。五個預先登記的 project seeds `[1,2,3,4,5]` 均在相同 code baseline `225cf8d44c36a8a210f6989bf76b9ebfe460adbd` 下執行 200 epochs、每 epoch 390 updates、合計 78,000 updates，並只採用 epoch-200 final checkpoint。五個 runs 均無 resume。

最終 test errors 為 4.41%、4.40%、4.32%、4.28%、4.43%；排序後為 4.28%、4.32%、4.40%、4.41%、4.43%，所以 frozen median 為 **4.40%**，對應 Seed 2。重現值比論文數值高 **0.13 percentage points**。專案沒有預先核准的容許誤差，因此只做透明數值比較，不以這個差異單獨宣告 performance success、failure、equivalence 或 statistical insignificance。

技術完整性方面，最後一次獨立稽核確認 45/45 formal artifact hashes、20/20 checkpoints 與 1,035/1,035 log events 均通過；模型 state、BatchNorm buffers、optimizer momentum、CPU/CUDA RNG、training cursor 與 LR metadata 均可驗證。故本專案的保守結論是：**在 frozen、pre-registered modern-framework protocol 之下，重現具有技術有效性與可稽核性；觀測到的五-run median test error 為 4.40%。**

## 2. Research question

本專案回答兩個不同層次的問題：

1. 能否把 WRN 論文、作者官方 Torch7 code、歷史 dependency 行為與必要的現代框架假設，整理成一份不混淆來源層級、可被測試與追溯的 frozen protocol？
2. 在同一份 frozen protocol 下執行五個預先登記 seeds，epoch-200 CIFAR-10 test-error median 是多少；它與論文的 4.27% 有何純數值差異？

第一題是 protocol fidelity 與工程可重現性問題；第二題才是 formal reproduction result。兩者不可因為結果接近或不接近論文數字而混為一談。

## 3. Source and evidence governance

本專案採用下列證據層級：

| Classification | 用途 | 例子 |
|---|---|---|
| `PAPER-SPECIFIED` | 論文直接陳述的模型、設定或結果 | arXiv v4 Table 5 的 4.27%、11.0M、median over 5 |
| `OFFICIAL-CODE-SPECIFIED` | 作者官方 repository 的明確行為 | depth-to-block formula、shortcut routing、byte-scale mean/std |
| `HISTORICAL-DEPENDENCY-BACKED` | 日期相容但非 paper-run lockfile 的 Torch dependency 行為 | BN defaults、FC reset、coupled decay equation |
| `DERIVED` | 由已支持輸入與明確公式計算 | 390 updates/epoch、78,000 total updates |
| `IMPLEMENTATION-ASSUMPTION` | 來源無法唯一決定而由人員預先核准的現代化選擇 | PyTorch port、project seeds、epoch-200 selection、determinism policy |
| `FORMAL-REPRODUCTION-RESULT` | 五個正式 runs 產生並通過完整性驗證的觀測值 | 4.40% frozen median |

主要來源依序為：本地 arXiv v4 PDF、`docs/evidence_table.md`、作者官方 WRN repository commit `ae6d0d0561484172790c7a63c8ce6ade5a5a2914`、歷史 dependency snapshots、frozen project documents、formal run artifacts，最後才是 derived calculations。

## 4. Formal target and freeze identity

| Field | Frozen value | Classification / locator |
|---|---|---|
| Paper | *Wide Residual Networks* | `PAPER-SPECIFIED`; Zagoruyko & Komodakis |
| Paper version | arXiv:1605.07146v4 | `PAPER-SPECIFIED` |
| Result family | Table 5, mean/std, no dropout | arXiv v4 PDF p. 8, Table 5/caption |
| Dataset | CIFAR-10 | paper target + approved project scope |
| Model | WRN-16-8 `B(3,3)` | arXiv v4 p. 4 Table 1; p. 8 Table 5 |
| Paper reference | 4.27% median test error | `PAPER-SPECIFIED`; p. 8 Table 5 |
| Paper parameter count | 11.0M | `PAPER-SPECIFIED`; rounded |
| Formal config | `configs/wrn16_8_arxiv_v4_frozen.yaml` | Frozen v1 |
| Frozen YAML SHA256 | `18EF6815727ADF6816132954721194626950F4B40BD2606A5B7E2297EAB959B3` | recomputed PASS |
| Formal code baseline | `225cf8d44c36a8a210f6989bf76b9ebfe460adbd` | Git history + all run manifests |

Phase history was reconstructed from `git log` rather than inferred:

| Phase | Commit | Commit message |
|---|---|---|
| Phase 0 | `af4f5fe1d9e643ff032c23eb40353c23eddd3ed8` | Freeze WRN-16-8 arXiv v4 reproduction specification |
| Phase 1 | `99753b8cf2a5457faed8ddb695793a7f38e99729` | Implement and validate WRN-16-8 architecture |
| Phase 2 | `f1ff88b532f4d77803b36e4f48fa1b407e01a1b7` | Implement and validate WRN CIFAR-10 data pipeline |
| Phase 3 | `803f09a7a410318a8aea79a428746d63fdc817a7` | Implement and validate WRN training engine semantics |
| Maintenance | `ee18d4c2ae54b41fb6f800a42c90c072f754355e` | Ignore local virtualenv in Phase 3 artifact boundary audit |
| Maintenance | `c4dd870b328bf8bc3c5d5aed35910935de227447` | Ignore local virtual environment |
| Phase 4 | `ef6363e0228fac51f70c77a79ab548e6fc60dc1f` | Validate WRN RTX 3070 Ti CUDA training environment |
| Phase 5 | `225cf8d44c36a8a210f6989bf76b9ebfe460adbd` | Freeze WRN formal training launch protocol |

## 5. WRN-16-8 architecture

`WRN-n-k` 中的 `n` 是 convolutional depth label，`k` 是 widening factor；這個記法是 `PAPER-SPECIFIED`。將 depth 轉為每個 group 的 block count 使用作者 code 的公式：

`N = (depth - 4) / 6 = (16 - 4) / 6 = 2`

此公式為 `OFFICIAL-CODE-SPECIFIED`，不是論文正文直接寫出的代數式。

| Stage | Spatial size | Channels | Blocks / operation |
|---|---:|---:|---|
| Input | 32x32 | 3 | CIFAR RGB |
| Stem | 32x32 | 16 | 3x3 conv, stride 1 |
| Group 1 | 32x32 | 128 | 2 x `B(3,3)` |
| Group 2 | 16x16 | 256 | 2 x `B(3,3)`, first block stride 2 |
| Group 3 | 8x8 | 512 | 2 x `B(3,3)`, first block stride 2 |
| Head | 1x1 | 512 -> 10 | BN, ReLU, global average pool, linear classifier |

Frozen block routing preserves pinned Torch7 semantics：

- residual branch 的第一個 3x3 convolution 接收 `ReLU(BN(x))`；
- dimension-preserving shortcut 使用 raw `x` identity；
- dimension-changing shortcut 使用對 `ReLU(BN(x))` 的 1x1 strided projection；
- residual addition 後沒有 post-add ReLU；下一個 block 自己從 BN/ReLU 開始。

Dimension-changing shortcut 的精確 tensor routing 是 `OFFICIAL-CODE-SPECIFIED`；論文圖與 Table 1 並未唯一決定這個細節。Formal target 的 dropout 為 0，且模型中沒有 active dropout module。

Phase 1 實際計得 trainable parameter count：

`10,961,370 = 10.96137M -> 11.0M`（以 one-decimal million 四捨五入）

10,961,370 是 implementation audit；11.0M 是論文 Table 5 的 rounded value。兩者相符但分類不同。

## 6. Data pipeline

Dataset artifact 是 CIFAR-10 official Python archive；project SHA256 為：

`6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE`

完整性檢查確認 50,000 train examples、10,000 test examples、10 classes。Paper p. 6 提供 split 數量與 moderate augmentation family；archive hash 是 project artifact identity，不是 paper checksum。

Normalization 保留 0..255 byte scale：

| Channel | Mean | Std |
|---|---:|---:|
| R | 125.3 | 63.0 |
| G | 123.0 | 62.1 |
| B | 113.9 | 66.7 |

Mean/std family 是 `PAPER-SPECIFIED`；精確 byte-scale values 是 `OFFICIAL-CODE-SPECIFIED`。

Formal train transform order：

1. horizontal flip；`p=0.5`，historical branch 中 draw 0 代表 flip；
2. four-sided reflection pad 4，edge-excluding；
3. random 32x32 crop，x/y offsets 分別均勻取自 0..8；
4. float32 conversion，保留 0..255 scale；
5. channel-wise normalization。

Paper 支持 flip、reflection padding、random crop 的 augmentation family；order、branch convention、offset domain 與 boundary mapping 則是 official-code／historical-dependency derived 的 frozen micro-semantics。

Data RNG 使用獨立 epoch seed：zero-based epoch `e0` 對應 `epoch_data_seed=e0+1`，所以 200 epochs 使用 seeds 1..200。sample augmentation seed 由 epoch seed 與 numeric sample index 決定，不依賴 worker scheduling；Windows `num_workers=0` 與 `2` 的 sample order、labels、transformed tensors 已驗證相同。這是 approved deterministic port，不宣稱重建未知的 Torch7 two-worker RNG stream。

## 7. Frozen training protocol

| Item | Frozen value | Provenance |
|---|---|---|
| Loss | `CrossEntropyLoss`, mean reduction, label smoothing 0 | paper + historical criterion cross-check |
| Batch size | 128 | `PAPER-SPECIFIED` |
| Train drop-last | true | official-code behavior, approved |
| Train batches/epoch | 390 | `DERIVED`: floor(50,000/128) |
| Presented train samples/epoch | 49,920 | `DERIVED`: 390x128 |
| Epochs | 200 | `PAPER-SPECIFIED` |
| Total updates | 78,000 | `DERIVED`: 390x200 |
| Optimizer | ordinary `torch.optim.SGD`, one parameter group | port implementation |
| Initial LR | 0.1 | paper/code |
| Momentum | 0.9 | arXiv v4/code |
| Dampening | 0 | arXiv v4/code |
| Nesterov | true | paper-era recipe/code |
| Weight decay | 0.0005, coupled | value paper-specified; equation/scope source-derived |
| Weight-decay scope | all trainable parameters | approved official/dependency-derived candidate |

Weight decay scope includes convolution weights、BN gamma/beta、FC weight 與 FC bias；convolution bias 不存在，BN running statistics 是 buffers 而非 parameters。這不是論文直接寫出的 scope，而是作者 code 的 flat parameter vector、historical `optim.sgd` scalar `dfdx += wd*x` 路徑與實際模型 parameter presence 的組合推導。

## 8. LR schedule and update accounting

LR 在 one-based epoch 開始、第一個 update 前套用：

| One-based epochs | LR | Derived global-update range |
|---|---:|---:|
| 1-59 | 0.1 | 1-23,010 |
| 60-119 | 0.02 | 23,011-46,410 |
| 120-159 | 0.004 | 46,411-62,010 |
| 160-200 | 0.0008 | 62,011-78,000 |

Epoch milestones 與 x0.2 factor 是 arXiv v4 所述；「start-of-epoch」由作者 code cross-check。Global-update ranges 是 `DERIVED` audit checks，不是 paper wording。

`global_update` 定義為成功完成的 `optimizer.step()` 數量。Non-finite loss/gradient 或 optimizer failure 不得推進計數；cursor invariant 為：

`global_update = (current_epoch_1 - 1) * 390 + next_batch_index_0`

## 9. Reproducibility engineering

本專案的可稽核性不是來自「設了 random seed」這個單一動作，而是跨 phase 的多層驗證：

- architecture depth、stage widths、block count、shortcut tensor routing 與 exact parameter count；
- synthetic forward/backward、finite loss 與 full gradient coverage；
- frozen initialization、same-seed state-dict replay；
- manual SGD first/second-step equation、coupled decay 與 Nesterov semantics；
- BN affine 與 FC bias 的 weight-decay coverage；
- LR boundary tests 與 exact global-update accounting；
- fail-closed checkpoint schema、model/optimizer/momentum/BN/cursor/LR restore；
- Python、NumPy、torch CPU、torch CUDA RNG capture/restore；
- mid-epoch and milestone resume；uninterrupted vs resumed exact equality；
- fresh-process CUDA deterministic replay；
- Windows zero-worker/two-worker data replay；
- RTX 3070 Ti target validation 與 CIFAR GPU forward-only preflight；
- formal result、checkpoint、log 與 artifact SHA manifests。

Deterministic execution 的意義是：在已定義、已支持的現代環境與同一 seed/protocol 下可以重播；它不保證與未知的 2016 Torch7/CUDA/cuDNN trajectory bitwise 相同，也不消除不同 run seeds 之間的結果差異。

## 10. Target hardware validation

Formal runs 綁定以下 target environment：

| Item | Actual evidence |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu126 |
| torchvision | 0.28.0+cu126 |
| NumPy | 2.4.4 |
| CUDA build | 12.6 |
| cuDNN | 91002 |
| NVIDIA driver | 591.86 |
| GPU | NVIDIA GeForce RTX 3070 Ti |
| VRAM | 8,589,410,304 bytes |
| Compute capability | 8.6 |
| Precision | FP32 |
| AMP / TF32 / compile | false / false / false |
| cuDNN benchmark | false |
| cuDNN deterministic | true |
| Deterministic algorithms | true; warn-only false |

Phase 4 在實際 GPU 上驗證 synthetic batch 128 forward/backward、full finite gradients、manual CUDA SGD update、CUDA RNG replay、checkpoint restore、uninterrupted/resumed exact equality、fresh-process replay 與真實 CIFAR batch forward-only path。CIFAR preflight 執行 0 backward、0 optimizer steps，未提前暴露 test metric。

## 11. Formal run governance

Formal execution在 Phase 5 commit `225cf8d...` 凍結後才開始。每個 run 的 manifest 在第一個 CIFAR optimizer step 前綁定：target、seed、code SHA、frozen hashes、dataset integrity、environment/package inventory、numerical policy、model/optimizer/schedule、checkpoint plan 與 initial training fingerprint。

Formal constraints：

- project seeds 固定為 `[1,2,3,4,5]`；它們不是 paper seeds；
- runs sequentially authorized and executed；
- after Seed 1 first step，不得依結果改 architecture、data、optimizer、schedule、evaluation 或 environment；
- checkpoints 在 epochs 60/120/160/200 後保存；
- per-run official result 只來自 epoch-200 final checkpoint；
- 不使用 best-test checkpoint、early stopping 或 post-hoc seed selection；
- completed run 不可覆寫或重新 resume。

## 12. Formal five-run results

下表直接來自各 run 的 `final_result.json`，seed-to-result mapping 未由排序值反推：

| Seed | Epochs | Updates | Resume | Correct | Incorrect | Accuracy | Test error | Epoch-200 checkpoint SHA256 | Technical status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 200 | 78,000 | 0 | 9,559 | 441 | 95.59% | 4.41% | `4AB151DEBC161CD719F0E33B7B3676246B52CC1F8BAE68DA01E62DC030601F47` | VALID |
| 2 | 200 | 78,000 | 0 | 9,560 | 440 | 95.60% | 4.40% | `B26BB0E52E2B10D808BE7B8BE724624D1DA0338A84BC9DD181FF8559FC2217FD` | VALID |
| 3 | 200 | 78,000 | 0 | 9,568 | 432 | 95.68% | 4.32% | `7D60439E946680B8EE794157817581BFB560E98075FC754BF6EACAA173984DEA` | VALID |
| 4 | 200 | 78,000 | 0 | 9,572 | 428 | 95.72% | 4.28% | `6BB022C606978E106337AF217B8EE7E512E62F5FD0578A52F9A40716A1D464F2` | VALID |
| 5 | 200 | 78,000 | 0 | 9,557 | 443 | 95.57% | 4.43% | `D0144AF2F2119DA2E2D8B4118A91AD1E9C6E47948CD5AA213B98BF5C67506FF1` | VALID |

最後一次完整稽核：

- formal runs: **5/5 VALID**；
- artifact manifest SHA checks: **45/45 PASS**；
- checkpoints loaded and validated: **20/20 PASS**；
- log events: **1,035/1,035 PASS**；
- failure / interruption / resume events: **0 / 0 / 0**；
- all runs: 200 epochs、78,000 updates、epoch-200-only result、0 resume。

## 13. Frozen median calculation

Frozen YAML 規定：exactly 5 runs、statistic `median`、input `epoch_200_cifar10_test_error`，且禁止 post-hoc seed or aggregation change。

按 test error 排序：

`4.28%, 4.32%, 4.40%, 4.41%, 4.43%`

中間第三個值為 **4.40%**，對應 Seed 2；其 accuracy 為 **95.60%**，9,560 correct、440 incorrect、10,000 total。

Median calculation canonical payload SHA256：

`993C9057DFF5B265DCE0E6087AB63628CC1DE2E272D0D2DC7D263D0DCD28C49D`

完整輸入與 digest 記錄於 `docs/formal_median_transcript.json`。該欄位是 canonical calculation payload 的 SHA；JSON 文件本身另有獨立 file SHA，兩者不可混稱。

## 14. Paper vs reproduction

| Item | Paper | Reproduction | Difference / note |
|---|---|---|---|
| Model | WRN-16-8 `B(3,3)` | WRN-16-8 `B(3,3)` | Target identity aligned |
| Dataset | CIFAR-10 | CIFAR-10 official Python archive | 50k train / 10k test |
| Preprocessing family | mean/std | byte-scale mean/std | Exact numbers from official code |
| Dropout | no dropout for Table 5 row | 0; no active module | aligned target family |
| Aggregation | median over 5 runs | median over seeds 1..5 | paper seeds unknown; project seeds pre-registered |
| Test error | 4.27% | **4.40%** | **+0.13 pp** |
| Parameter count | 11.0M rounded | 10,961,370 = 10.96137M -> 11.0M | rounded agreement |
| Framework | Torch7-era implementation | PyTorch 2.13.0+cu126 port | known framework difference |
| Hardware/runtime | exact paper-run lock unknown | RTX 3070 Ti, CUDA 12.6, cuDNN 91002 | modern explicit lock |
| Training provenance | paper + author code | frozen synthesis + approved assumptions | classifications preserved |

本比較不宣稱 statistical equivalence，也不把 0.13 pp 轉換成未經需求的 relative percentage。

## 15. Difference analysis

### Known differences

- 現代 PyTorch port，而不是原 paper-era Torch7 ecosystem；
- CUDA 12.6、cuDNN 91002 與 RTX 3070 Ti，而非未完整鎖定的 paper runtime；
- project seeds `[1,2,3,4,5]` 不宣稱等於論文五個 seeds；
- paper 未唯一說明的 checkpoint selection、FC/BN defaults、worker RNG stream 與若干 micro-semantics，以官方 code、歷史 dependency evidence 和預先核准 assumptions 凍結。

### Possible contributors, not proven causes

- floating-point implementation 與 kernel ordering differences；
- library kernel、compiler/runtime 與 CUDA/cuDNN version differences；
- 未公開或未保存的原始環境細節；
- 正常的 stochastic run distribution。

沒有證據能把 **+0.13 pp** 歸因於任何單一因素；「implementation 一定錯」或「差異一定只是隨機」都超出現有證據。

## 16. Limitations

1. exact paper seeds、paper checkpoint-selection rule 與完整 paper-run dependency lockfile 仍為 `UNKNOWN`；
2. deterministic PyTorch replay 不等於 bitwise Torch7 historical replay；
3. formal study 只涵蓋 CIFAR-10 WRN-16-8 mean/std no-dropout family；
4. 五個 project seeds 足以遵守 frozen aggregation count，但不足以支持額外的統計等價性推論；
5. 沒有預先註冊 success tolerance 或 hypothesis test，故不能由 +0.13 pp 推導 significance 或 pass/fail；
6. historical dependency evidence 是 date-compatible source anchors，不是 recovered paper-run installation record。

## 17. Technical validity conclusion

本重現的技術有效性建立在 frozen protocol fidelity 與 artifact integrity，而不是把結果調到等於 4.27%。所有五個 runs 均在同一 baseline、同一 environment policy、同一 model/data/optimizer/schedule 下完成，且完整 artifacts 通過獨立驗證。

因此可安全陳述：

> The reproduction is technically valid under the frozen, pre-registered modern-framework protocol. All five runs and formal artifacts passed integrity validation. The observed five-run median CIFAR-10 test error was 4.40%, which is numerically 0.13 percentage points above the paper's reported 4.27% median. Because no tolerance was pre-registered, the difference is reported transparently without retroactive performance success/failure classification.

不能安全陳述：「完全複製 4.27%」、「與論文統計等價」、「0.13% 不顯著」或「implementation 已被結果數字證明完全正確」。

## 18. Reproduction lessons learned and next work

最重要的學習不是單一 accuracy 數字，而是把「論文說了什麼」、「作者 code 做了什麼」、「歷史 dependency 可能提供什麼」、「現代 port 必須自己決定什麼」拆開。這使 shortcut routing、weight-decay scope、BN/FC initialization、data RNG 與 checkpoint selection 等容易被現代慣例悄悄改寫的細節，變成可被教授追問、可由測試回答的明確主張。

另一個核心學習是 no-post-hoc tuning：知道 paper value 後，重現很容易變成追逐 4.27%。本專案在第一次正式 optimizer step 前凍結 protocol，之後接受觀測到的 4.40%，避免用 test result 反向調整設定。這比得到表面更接近的數字更符合可稽核研究實務。

下一個合理研究方向是 DenseNet：它延續 feature reuse 與 gradient-flow 的架構問題，但其 dense connectivity、growth rate 與 transition layers 會帶來不同的 topology、memory 與 evidence-audit 挑戰。後續研究應建立新的獨立 freeze，而不是沿用 WRN assumptions。

## Appendix A. Professor-facing 30-second summary

我重現的是 Zagoruyko 與 Komodakis 的 Wide Residual Networks，正式目標是 arXiv v4 Table 5 的 CIFAR-10 WRN-16-8、mean/std、no-dropout 設定。我先把論文、官方 Torch7 code、歷史 dependency 與現代 PyTorch assumptions 分類並凍結，再用同一 RTX 3070 Ti baseline 跑五個預先登記 seeds。五次都完成 200 epochs、78,000 updates、0 resume，45 個正式 artifact hashes 與 20 個 checkpoints 全部通過。Frozen median test error 是 4.40%，論文是 4.27%，差 +0.13 percentage points；因為沒有預先設定 tolerance，我只做透明比較，不事後調參或宣告統計等價。

## Appendix B. Professor-facing 2-minute summary

這個專案重現 *Wide Residual Networks*，但不是泛稱「做一個 WRN」。我把 formal target 精確綁定到 arXiv:1605.07146v4 Table 5：CIFAR-10、WRN-16-8、`B(3,3)`、mean/std normalization、no dropout、五-run median reference 4.27%。架構是 depth 16、widen factor 8，每組兩個 residual blocks，channel widths 16、128、256、512；實作的 exact trainable parameter count 是 10,961,370，四捨五入後與 paper 11.0M 一致。

研究方法上，我不把所有設定都說成 paper facts。4.27%、median over five、batch 128、200 epochs 是 paper evidence；shortcut tensor routing 與 byte-scale statistics 來自作者官方 code；BN/FC defaults 與 coupled decay equation需要歷史 dependency evidence；PyTorch port、project seeds、epoch-200 checkpoint selection 與 deterministic FP32 policy 則是預先核准的 implementation assumptions。

在正式執行前，我鎖定 code baseline、frozen YAML SHA、dataset SHA、RTX 3070 Ti environment 與 no-post-seed1-tuning rule。Seeds 1 到 5 都獨立完成 200 epochs、78,000 updates、0 resume，只取 epoch-200 result。五個 errors 是 4.41、4.40、4.32、4.28、4.43%，排序後中位數是 4.40%。最終 audit 驗證 45/45 formal hashes、20/20 checkpoints、1,035/1,035 events。

所以結論是：這是一個在 frozen modern-framework protocol 下技術有效、可稽核的重現；觀測中位數比 paper 高 0.13 percentage points。由於沒有預先註冊 tolerance，我不把這個差異包裝成成功、失敗或不顯著，也不為了追到 4.27% 事後調參。
