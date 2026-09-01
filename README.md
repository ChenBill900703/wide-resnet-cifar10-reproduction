# Wide Residual Networks：CIFAR-10 可稽核重現

這個專案不是只把 WRN 模型「跑起來」，而是把論文、作者官方程式碼、現代 PyTorch 實作、五次正式訓練與全部結果檔案串成一條可以重新檢查的證據鏈。

## 一眼看懂結果

| 項目 | 原論文 | 本專案 | 差距 |
|---|---:|---:|---:|
| CIFAR-10 test error（五次中位數） | **4.27%** | **4.40%** | **+0.13 percentage points** |
| 對應 accuracy | 95.73% | 95.60% | -0.13 percentage points |
| 模型參數量 | 11.0M（論文四捨五入） | 10,961,370 → 11.0M | 四捨五入一致 |

`+0.13 percentage points` 的意思是：以 10,000 張 CIFAR-10 test images 來看，中位數結果約多錯 13 張。

這個差距只做透明比較。因為論文沒有公開五次個別結果與 seed，也沒有事先設定統計等價界線，所以本專案不把 0.13 pp 包裝成「完全重現」、「統計等價」或「重現失敗」。

## 正式重現目標

- 論文：Zagoruyko & Komodakis, *Wide Residual Networks*
- 版本：arXiv:1605.07146v4
- 位置：第 8 頁 Table 5
- Dataset：CIFAR-10
- Model：WRN-16-8，`B(3,3)`，widen factor `k=8`
- Preprocessing：mean/std normalization
- Augmentation：horizontal flip + reflection padding + random crop
- Dropout：0
- Batch size：128
- Training：200 epochs，SGD + Nesterov
- 正式統計量：五個預先登記 runs 的 epoch-200 test-error median

> 注意：論文 Table 4 的 WRN-16-8 4.81% 使用 ZCA preprocessing，與本專案選定的 Table 5 mean/std 4.27% 不是同一個比較條件。

## 五次正式結果

| Seed | Correct / 10,000 | Test error | 狀態 |
|---:|---:|---:|---|
| 1 | 9,559 | 4.41% | VALID |
| 2 | 9,560 | **4.40%** | VALID / MEDIAN |
| 3 | 9,568 | 4.32% | VALID |
| 4 | 9,572 | 4.28% | VALID |
| 5 | 9,557 | 4.43% | VALID |

排序後為 `4.28, 4.32, 4.40, 4.41, 4.43`，所以預先指定的中位數是 **4.40%**。不能改挑表現最好的 Seed 4，因為正式規則在訓練前已經凍結為 median。

## 這個 repository 包含什麼

| 路徑 | 內容 |
|---|---|
| `src/wrn/` | WRN 模型、初始化、資料處理、optimizer、LR schedule、checkpoint、RNG 與 formal launcher |
| `tests/` | architecture、data、optimizer、resume、CUDA 與 formal protocol 測試 |
| `configs/` | 已凍結的 WRN-16-8 正式設定 |
| `docs/` | evidence table、assumptions、各 phase 驗證報告與最終分析 |
| `references/` | 專案使用的論文 PDF |
| `data/` | CIFAR-10 archive 與解壓後的 Python batches |
| `wide-resnet-formal-runs/` | Seeds 1–5 的 manifests、logs、results 與 20 個 checkpoints |
| `WRN_CIFAR10_Reproduction_Final.pptx` | 最終答辯簡報 |
| `SHA256SUMS.txt` | 發佈檔案的 SHA-256 完整性清單 |

## 證據怎麼分級

專案刻意不把所有設定都說成「論文寫的」：

1. `PAPER-SPECIFIED`：論文直接寫出的架構、訓練設定或結果。
2. `OFFICIAL-CODE-SPECIFIED`：作者官方 repository 的實際行為。
3. `HISTORICAL-DEPENDENCY-BACKED`：paper-era Torch dependency 可追溯的預設行為。
4. `DERIVED`：由已知輸入和完整公式計算出的數字。
5. `IMPLEMENTATION-ASSUMPTION`：來源無法唯一決定、但在訓練前明確凍結的現代實作選擇。
6. `FORMAL-REPRODUCTION-RESULT`：五次正式執行實際觀測到的結果。

完整來源與 locator 請看 [`docs/evidence_table.md`](docs/evidence_table.md)，正式規格請看 [`docs/reproduction_spec.md`](docs/reproduction_spec.md)，最終比較請看 [`docs/final_reproduction_report.md`](docs/final_reproduction_report.md)。

## 下載完整資料

這個 repository 使用 **Git LFS** 保存 CIFAR-10 archive、解壓後 batches 與 checkpoints；完整內容約 2 GiB。請先安裝 Git LFS：

```bash
git lfs install
git clone https://github.com/ChenBill900703/wide-resnet-cifar10-reproduction.git
cd wide-resnet-cifar10-reproduction
git lfs pull
```

若只下載一般 Git 檔案而沒有執行 `git lfs pull`，大型資料會顯示成很小的 LFS pointer，並不是實際 checkpoint。

## 驗證下載內容

`SHA256SUMS.txt` 記錄的是實際檔案內容的 SHA-256，不是 Git LFS pointer 的 hash。完成 `git lfs pull` 後，可用 PowerShell 抽查：

```powershell
Get-FileHash -Algorithm SHA256 `
  .\configs\wrn16_8_arxiv_v4_frozen.yaml, `
  .\references\wide_residual_networks.pdf, `
  .\wide-resnet-formal-runs\WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1\seed_02\final_result.json
```

正式五次中位數的完整輸入、排序與 digest 位於 [`docs/formal_median_transcript.json`](docs/formal_median_transcript.json)。

## 重要限制

- 本專案是現代 PyTorch port，不是 2016 Torch7 runtime 的 bitwise replica。
- 原論文五次執行的 seed identities、個別結果與完整 dependency lock 未公開。
- 五次 runs 足以遵守論文的 aggregation count，但不足以單獨主張統計等價。
- `4.40%` 是預先凍結 protocol 下的正式觀測結果；沒有為了追到 `4.27%` 事後調參。

## 最保守、也最準確的結論

> 本專案完成了一個可追溯、可驗證的 WRN-16-8 CIFAR-10 modern-framework reproduction。五次正式執行的 frozen median test error 是 4.40%，比原論文 4.27% 高 0.13 percentage points。所有差異如實保留，不以事後調參掩蓋。
