# WRN CIFAR-10 Professor Defense Q&A

使用原則：先答「短答」，教授追問時才展開「深入追問」。所有 performance 結論都限於 frozen protocol；不可把 4.40% 說成完全等於、證明或否定論文的 4.27%。

## A. Paper understanding

### Q1. 為什麼選 WRN-16-8，而不是其他 WRN？

**短答（20–40 秒）**：因為正式 target 不是任意挑一個型號，而是綁定 arXiv v4 Table 5 的 WRN-16-8、CIFAR-10、mean/std、no-dropout 結果家族。它有明確的 4.27% 五-run median 與 11.0M 參數量，可形成可稽核的 comparison target。

**深入追問**：其他 WRN 可能有更高 accuracy，但會同時改變 depth、width、dropout 或 preprocessing，無法回答同一個研究問題。專案在 Phase 0 就把 target、paper version、run count、aggregation 與 checkpoint rule 凍結，避免看到結果後換 target。

**證據**：`references/wide_residual_networks.pdf` p.8 Table 5；`docs/reproduction_spec.md` §1–2；`configs/wrn16_8_arxiv_v4_frozen.yaml`。

### Q2. 論文的 4.27% 是什麼？為什麼不是你自己推算的？

**短答（20–40 秒）**：4.27% 是 arXiv:1605.07146v4 第 8 頁 Table 5 的 WRN-16-8 CIFAR-10 mean/std no-dropout test error，分類是 `PAPER-SPECIFIED`，不是從 accuracy 換算或從第三方網站抄來的。

**深入追問**：我同時以 PDF 視覺檢查表格與文字擷取交叉確認，並保留 paper SHA。這個數字只能在該模型、資料處理與 aggregation 上下文使用，不能說成「WRN-16-8 永遠是 4.27%」。

**證據**：`references/wide_residual_networks.pdf` p.8 Table 5；`docs/evidence_table.md` CIFAR-10 reported errors rows。

### Q3. 4.27% 是 single run 還是 median？

**短答（20–40 秒）**：是五次執行的 median。Table 5 caption 明確說結果是 five runs 的 medians，因此本專案也在訓練前凍結五個 runs 與 median aggregation。

**深入追問**：論文沒有公開那五次的 individual seeds 或 individual errors，所以我們只能重現相同的 run count 與 aggregation rule，不能宣稱 seed identity 相同。

**證據**：paper p.8 Table 5 caption；`docs/reproduction_spec.md` evaluation/aggregation sections。

## B. Architecture

### Q4. 「16-8」代表什麼？

**短答（20–40 秒）**：16 是 CIFAR WRN 的 nominal depth，8 是 widening factor `k`。基礎 stage widths 是 16、16k、32k、64k，所以 k=8 時 residual groups 是 128、256、512 channels。

**深入追問**：depth 與 width 必須一起說明；只講 WRN-16-8 還不夠，正式 target 另外綁定 `B(3,3)` block、dropout 0、資料處理與訓練 protocol。

**證據**：paper pp.4–5, Table 1 and §3；`docs/reproduction_spec.md` architecture section。

### Q5. 為什麼 `N=(16-4)/6=2`？

**短答（20–40 秒）**：CIFAR WRN 的 depth relation 是 `n=6N+4`。把 n=16 代入，得到 N=2，也就是三個 residual groups 各有兩個 `B(3,3)` blocks。

**深入追問**：這是 architecture-specific relation，不應外推到所有 ResNet。`6N` 來自三個 groups、每個 block 兩個 convolution；其餘 4 對應 stem、尾端與分類結構的計數慣例。

**證據**：paper §3；`docs/evidence_table.md` architecture depth formula/block count rows；Phase 1 architecture tests。

### Q6. 你實作的完整 stage 是什麼？

**短答（20–40 秒）**：32×32 input 先進入 16-channel stem；Group 1 是 128 channels、2 blocks、32×32；Group 2 是 256、2 blocks、16×16；Group 3 是 512、2 blocks、8×8；最後 BN/ReLU、global average pooling 與 10-class classifier。

**深入追問**：dimension-changing group entry 依 frozen shortcut routing 處理。pre-activation 與 shortcut 的細節來自 official Torch7 semantics 和專案測試，不把它們誤標為 Table 5 的逐字說明。

**證據**：paper p.4 Table 1；`docs/reproduction_spec.md`；`docs/phase1_architecture_validation.md`；`src/wrn/model.py`。

### Q7. 參數量怎麼驗證？

**短答（20–40 秒）**：本實作逐 tensor 計數得到 10,961,370 trainable parameters，也就是 10.96137M；依論文一位小數 million 的粒度四捨五入為 11.0M，和 Table 5 相符。

**深入追問**：一致的 rounded parameter count 是 architecture cross-check，不足以證明所有 layer ordering、initialization 或 training semantics 完全相同，所以另有 shape、forward/backward、shortcut 和 initialization tests。

**證據**：paper p.8 Table 5；`docs/phase1_architecture_validation.md`；architecture test suite。

## C. Why width instead of depth

### Q8. 為什麼 wider network 可以勝過更深的 network？

**短答（20–40 秒）**：WRN 論文的核心觀察是，極深 residual networks 的每層 feature 可能很窄，增加 depth 會帶來序列計算與 diminishing feature reuse。增加 width 能提高每個 block 的 representational capacity，並在 GPU 上提供更多平行度。

**深入追問**：這是論文在其實驗範圍內的 argument，不是「width 永遠優於 depth」的普遍定理。WRN 仍利用 residual connections，重點是調整 capacity 的配置方式。

**證據**：paper abstract、§1、§3–4。

### Q9. WRN 和 ResNet 的關係是什麼？

**短答（20–40 秒）**：WRN 保留 residual learning 與 shortcut 的骨架，但以 widening factor 擴大 residual groups，目的不是丟掉 ResNet，而是改善只靠增加 depth 的效率與能力配置。

**深入追問**：因此比較時要把 block family、depth counting 與 shortcut semantics 說清楚；不能只把 WRN 描述成「比較多 channels 的普通 CNN」。

**證據**：paper §2–3；`docs/evidence_table.md` architecture rows。

## D. Data pipeline

### Q10. Data preprocessing 實際做了什麼？

**短答（20–40 秒）**：輸入保留 byte scale 0–255，channel mean 是 125.3、123.0、113.9，std 是 63.0、62.1、66.7。訓練順序是 horizontal flip、reflection pad 4、random 32×32 crop，再 normalization。

**深入追問**：mean/std family 有 paper evidence；數值、byte-scale 與 transform ordering 等 micro-semantics 依 official code 與 frozen assumptions 分類。這是 evidence separation 的例子，不能全部說成 paper-specified。

**證據**：paper p.6 §4.2 and p.8 Table 5；`docs/evidence_table.md` preprocessing/augmentation rows；`docs/reproduction_spec.md` data section。

### Q11. 為什麼每個 epoch 是 390 batches？

**短答（20–40 秒）**：train set 50,000 張、batch size 128、`drop_last=true`，所以完整 batches 是 `floor(50000/128)=390`。這是 `DERIVED` cross-check，不是論文直接寫「390」。

**深入追問**：每 epoch 因此呈現 390×128=49,920 samples，剩餘 80 個位置在該 epoch 被 drop。由於每 epoch permutation 隨 epoch seed 重建，被省略的不是永久固定同一批樣本。

**證據**：frozen config；`docs/reproduction_spec.md`；Phase 2 sampler/DataLoader tests。

### Q12. 只呈現 49,920/50,000 會不會 bias training？

**短答（20–40 秒）**：`drop_last` 會讓每個 epoch 少 80 個 sample positions，但 permutation 每 epoch 改變，因此不是固定排除某 80 張。它可能影響 optimization semantics，所以專案把它預先凍結並測試，而不是把它當作無關細節。

**深入追問**：我沒有證據聲稱它完全沒有統計影響，也沒有在看到結果後改成 391 batches。科學上正確做法是揭露並保持 protocol，不做 post-hoc 調整。

**證據**：`configs/wrn16_8_arxiv_v4_frozen.yaml`；`docs/assumptions.md`；Phase 2 validation tests。

### Q13. Flip 與 crop 的 randomness 如何控制？

**短答（20–40 秒）**：flip probability 是 0.5；crop 的 x/y offsets 各為 0..8。正式 pipeline 以 one-based epoch number 建立 epoch data seed，讓 data order 與 augmentation 可重播。

**深入追問**：這種 seed mapping 是 project-defined deterministic contract，不是對 paper original RNG identity 的宣稱。它的價值是讓本專案的相同 seed/run 可稽核和 resume。

**證據**：`docs/reproduction_spec.md` data/RNG sections；Phase 2 replay tests；frozen config。

## E. Optimizer / LR

### Q14. Optimizer protocol 是什麼？

**短答（20–40 秒）**：SGD，initial LR 0.1、momentum 0.9、dampening 0、Nesterov true、weight decay 0.0005。loss 是 mean-reduction CrossEntropyLoss，label smoothing 0。

**深入追問**：這些不是全部來自單一來源；evidence table 分開記錄 paper、official code、historical dependency 與 assumptions，最後才進入 frozen config。

**證據**：`docs/evidence_table.md` optimizer rows；`configs/wrn16_8_arxiv_v4_frozen.yaml`；`docs/reproduction_spec.md` training section。

### Q15. 為什麼 BN gamma/beta 和 FC bias 也做 weight decay？

**短答（20–40 秒）**：因為 frozen protocol 的 weight-decay scope 是 all trainable parameters；這不是套用現代常見的 no-decay parameter groups。專案依 official/historical semantics 明確鎖定並用 analytical optimizer tests 驗證。

**深入追問**：是否該排除 BN 或 bias 是另一個可研究的 recipe，但如果正式跑完後為接近 4.27% 才改 scope，就會破壞 pre-registration。這裡報告的是凍結行為，不聲稱它是所有現代 WRN 的最佳做法。

**證據**：`docs/evidence_table.md` weight-decay scope rows；`docs/reproduction_spec.md`；Phase 3 coupled-decay tests。

### Q16. 什麼是 coupled weight decay？

**短答（20–40 秒）**：coupled decay 是先把 `λw` 加進 gradient，再進入 momentum/Nesterov 更新，而不是像 decoupled weight decay 那樣在 optimizer update 外獨立縮放權重。兩者在有 momentum 時不是同一個演算法。

**深入追問**：專案用可手算的一、二步 parameter update 比對實作，確認 decay、momentum buffer 與 Nesterov 的運算順序，而不是只檢查 optimizer 類別名稱。

**證據**：historical dependency evidence in `docs/evidence_table.md`；Phase 3 analytical optimizer tests；`src/wrn/training.py`。

### Q17. 如何驗證 Nesterov semantics？

**短答（20–40 秒）**：建立小型、固定初值與 gradient 的測試，手算第一步和第二步，包括 coupled decay、momentum buffer 初始化與 Nesterov look-ahead，再逐值比對 optimizer output。

**深入追問**：第二步很重要，因為只測第一步可能看不出 momentum accumulation 的錯誤。這也防止把不同 framework 同名參數誤認為完全相同 semantics。

**證據**：`docs/phase3_training_engine_validation.md`；Phase 3 optimizer semantic tests。

### Q18. LR schedule 和 78,000 updates 怎麼對上？

**短答（20–40 秒）**：每 epoch 390 updates，200 epochs 共 78,000。LR 在 one-based epoch 開始時套用：1–59 為 .1、60–119 為 .02、120–159 為 .004、160–200 為 .0008。

**深入追問**：換成 update ranges 是 1–23,010、23,011–46,410、46,411–62,010、62,011–78,000；這些 ranges 是由 390 updates/epoch 推得的 `DERIVED` cross-check，不冒充 paper wording。

**證據**：frozen config；`docs/reproduction_spec.md` LR section；Phase 3 boundary tests；formal logs。

### Q19. 如何驗證 epochs 60/120/160 的 boundary 沒有 off-by-one？

**短答（20–40 秒）**：測試直接檢查 boundary 前一個 epoch、boundary epoch 與對應 global update，並在 formal logs 核對 schedule events。LR 是在該 one-based epoch 開始前更新。

**深入追問**：除了 unit tests，五個 formal runs 的 log event sequence 也被完整 audit；1,035/1,035 events 通過，包含相同的 LR transitions 與 cursor progression。

**證據**：Phase 3 LR boundary tests；five formal logs；final audit summary in `docs/final_reproduction_report.md`。

## F. BatchNorm / initialization

### Q20. BatchNorm 與 initialization 是怎麼決定的？

**短答（20–40 秒）**：paper 沒有唯一決定所有框架層級細節，因此專案優先讀 author Torch7 implementation，再用 historical dependency source 還原當時 module defaults；仍無法唯一決定的部分才標為 human-approved implementation assumption。

**深入追問**：這避免把 later PyTorch defaults 倒填成 paper fact。Phase 1 另外測試 convolution、BN、classifier 的 initialization 與 state layout。

**證據**：`docs/evidence_table.md` initialization/BN rows；`docs/assumptions.md`；`docs/phase1_architecture_validation.md`。

### Q21. BN running state 為何是 checkpoint integrity 的一部分？

**短答（20–40 秒）**：BN running mean/variance 會影響 evaluation；只存 model weights 卻漏掉 buffers，就不能還原同一個 classifier state。因此 checkpoint audit 會檢查 BN buffers 存在、finite，variance 非負，並驗證 restore。

**深入追問**：最終 audit 在 20/20 checkpoints 都檢查 83 個 model-state tensors，包含 BN buffers；這是技術有效性證據，不是 performance equivalence 證據。

**證據**：checkpoint schema/restore tests；formal checkpoint audit；`docs/final_reproduction_report.md` §8、§12。

## G. Reproducibility

### Q22. 你怎麼證明這個 reproduction 可稽核？

**短答（20–40 秒）**：證據鏈從 paper locator、pinned official code、historical dependency、frozen config/SHA，一直到每個 run 的 environment、log、checkpoints、result 與 hash manifest。最後又獨立驗證 45/45 artifacts、20/20 checkpoints、1,035/1,035 events。

**深入追問**：可稽核不等於宣稱 paper-identical；它表示別人能定位每個重要決策、驗證 formal baseline、重算 median，並檢查 artifacts 未被替換。

**證據**：`docs/freeze_manifest.md`；phase validation docs；formal manifests；`docs/formal_median_transcript.json`。

### Q23. 什麼會讓一個 run 無效？

**短答（20–40 秒）**：baseline/config/dataset hash 不符、seed 不符、不是 200 epochs/78,000 updates、resume 規則不符、checkpoint selection 改變、缺 artifact、hash mismatch、log/cursor/LR 不連續、model/optimizer/RNG state 無效，都會讓 run 不能進入 frozen aggregation。

**深入追問**：規則在看五個結果前已存在；不是根據某 seed 表現好壞決定。這防止挑 seed、挑 checkpoint 或靜默修補 artifact。

**證據**：Phase 5 launcher/audit tests；formal manifests/hash manifests；`docs/reproduction_spec.md` invalidation rules。

## H. Why PyTorch instead of Torch7

### Q24. 為什麼用 PyTorch，不直接跑 Torch7？

**短答（20–40 秒）**：目標是用可維護、可測試的現代環境做 evidence-grounded reproduction。Torch7 paper-era ecosystem 難以在目前硬體與 dependency 上完整重建；因此 PyTorch port 被明確標為 implementation choice，而 training semantics 優先追溯 author Torch7 code。

**深入追問**：這不是假裝框架相同。專案把 paper claim、official-code semantics、historical defaults 與 port assumptions 分開，並以 unit/analytical/replay tests降低 translation risk。

**證據**：`docs/reproduction_spec.md` scope；`docs/evidence_table.md` official lineage；`docs/assumptions.md`。

### Q25. 現代 PyTorch 不同，這不就只是 reimplementation？

**短答（20–40 秒）**：它確實是現代框架 reimplementation，但同時是 protocol-faithful reproduction：target、evidence、assumptions、run count、aggregation 與結果選擇在訓練前凍結。不能稱為 bitwise replication，但可以評估相同 evidence-grounded protocol 在現代框架的結果。

**深入追問**：reimplementation 與 reproduction 不互斥；關鍵是範圍聲明。若宣稱 paper-era executable identity 就不成立，因此本專案刻意用 scoped wording。

**證據**：`docs/final_claim_matrix.md`；`docs/final_reproduction_report.md` conclusions/limitations。

## I. Modern CUDA environment

### Q26. PyTorch/CUDA/cuDNN 版本為什麼重要？

**短答（20–40 秒）**：framework 與 GPU libraries 決定 convolution kernels、floating-point reduction order、determinism支援與 RNG 行為，可能造成數值軌跡差異，所以 environment 是 formal evidence，而不是一般系統資訊。

**深入追問**：本次記錄 Python 3.11.9、PyTorch 2.13.0+cu126、CUDA 12.6、cuDNN 91002、driver 591.86 與 RTX 3070 Ti。這讓結果可定位，但不能倒推原作者環境。

**證據**：five run manifests；`docs/phase4_target_hardware_validation.md`；`docs/phase5_formal_run_freeze.md`。

### Q27. 為什麼選 RTX 3070 Ti？

**短答（20–40 秒）**：它是本專案可用且經 Phase 4 驗證的 target hardware，不是論文設定。專案驗證 device identity、compute capability 8.6、記憶體、FP32 policy 與 CIFAR GPU preflight，然後在同一環境跑五個 runs。

**深入追問**：不能說 RTX 3070 Ti 最符合 paper；正確說法是「本次現代環境已完整記錄並固定」。

**證據**：`docs/phase4_target_hardware_validation.md`；formal environment snapshots。

## J. Seeds / median

### Q28. 為什麼使用五個 seeds？

**短答（20–40 秒）**：paper Table 5 的 aggregation 是 five-run median，因此正式 protocol 也使用五個預先登記 project seeds，讓 aggregation 結構可比較。

**深入追問**：run count 有 paper evidence；具體 `[1,2,3,4,5]` 是 human-approved implementation assumption。五個 seeds 不是「足以代表所有 stochastic variation」的普遍保證。

**證據**：paper p.8 Table 5 caption；frozen config；`docs/assumptions.md`。

### Q29. 這些是論文原本的 seeds 嗎？

**短答（20–40 秒）**：不能這樣說。paper 與已稽核 official code 沒有唯一揭露 Table 5 五次 run 的 seed identities；本專案 seeds 1–5 是明確揭露、預先核准的 project seeds。

**深入追問**：把 unknown 說成相同會是假證據。這也是與 paper exact execution 不同的限制之一。

**證據**：`docs/open_questions.md` seed identity item；`docs/assumptions.md`；frozen config。

### Q30. 為什麼用 median，不用 mean？

**短答（20–40 秒）**：因為 formal comparison target 的 paper table 明確以 five-run median 報告。選 median 不是因為本次 median 比 mean 好看，而是在訓練前為對齊 paper aggregation 而凍結。

**深入追問**：若另報 mean 可作 descriptive supplement，但不能替換 frozen statistic。最終正式結果由排序後第三個值 4.40% 決定。

**證據**：paper Table 5 caption；`docs/reproduction_spec.md` aggregation rule；`docs/formal_median_transcript.json`。

## K. Why 4.40 instead of 4.27

### Q31. 為什麼你的結果是 4.40%，不是 4.27%？

**短答（20–40 秒）**：五個 formal errors 是 4.41、4.40、4.32、4.28、4.43%，排序中央值就是 4.40%。與論文不同的已知背景包括現代 PyTorch/CUDA stack、project seeds 並非 paper seeds，以及部分 micro-semantics 必須由 official code、historical dependencies或 assumptions 凍結。

**深入追問**：這些只是 known differences 或 possible contributors；沒有 controlled ablation，所以不能把 +0.13 pp 歸因給其中某一項。

**證據**：five `final_result.json`；`docs/formal_median_transcript.json`；final report difference analysis。

### Q32. 為什麼不調參到 4.27%？

**短答（20–40 秒）**：reproduction 應測試預先登記的 protocol，而不是知道答案後倒推設定。protocol 凍結後，五個 seeds 都不變地執行，4.40% 就是應接受並透明報告的觀測結果；事後 tuning 反而會降低證據價值。

**深入追問**：若要研究另一個 recipe，應建立新版本、重新說明 evidence/assumptions並在看新結果前凍結；不能覆寫本次 formal baseline。

**證據**：Git freeze history；frozen config SHA；formal baseline `225cf8d...`；five manifests。

### Q33. 能不能宣稱「成功重現」？

**短答（20–40 秒）**：可以在限定範圍說「technically valid, protocol-faithful reproduction」，因為 frozen protocol 與 artifacts 經完整稽核；但不能說「完全重現 4.27%」或僅憑 +0.13 pp 宣告 performance pass。

**深入追問**：專案沒有事前 performance tolerance，也沒有等價性統計設計，因此結論分成 technical validity 與 transparent numerical comparison。

**證據**：`docs/final_claim_matrix.md` wording lock；final report §§13–16。

## L. Statistical interpretation

### Q34. +0.13 pp 顯著嗎？

**短答（20–40 秒）**：目前不能回答顯著或不顯著。+0.13 是 percentage-point 的描述性差值；專案沒有預註冊 significance test、equivalence margin 或足夠的分布假設，所以不做事後統計判定。

**深入追問**：若要做 inferential claim，需事先設計樣本數、estimand、test 或 equivalence margin，並考慮 paper 只有 aggregate median、沒有 individual-run data 的限制。

**證據**：`docs/reproduction_spec.md`；`docs/open_questions.md`；final report statistical interpretation。

### Q35. 五個 seeds 科學上夠嗎？

**短答（20–40 秒）**：五個 runs 足以忠實實施 paper 的 median aggregation 規模，但不足以支持廣泛的 variance 或 equivalence 結論。這裡的主張是 protocol comparison，不是完整估計 stochastic performance distribution。

**深入追問**：更多 seeds 能提高分布估計能力，但若新增到正式 aggregate，必須是另一個預先定義的研究，不可事後改寫 frozen five-run result。

**證據**：paper Table 5 caption；frozen aggregation rule；final report limitations。

## M. Checkpoint / resume

### Q36. 你怎麼知道 resume 是 exact？

**短答（20–40 秒）**：測試把 uninterrupted trajectory 與 mid-epoch checkpoint/resume trajectory 做 exact equality 比對，restore model、optimizer/momentum、CPU/CUDA RNG、BN buffers 與 cursor。formal runs 本身的 resume count 都是 0，但 resume engine 仍被驗證。

**深入追問**：只 restore weights 不夠；若 sampler position、next batch index、RNG 或 momentum 遺失，後續 trajectory 就會分歧。checkpoint schema 對這些欄位都有檢查。

**證據**：`docs/phase3_training_engine_validation.md`；Phase 4 fresh-process CUDA replay；checkpoint schema tests。

### Q37. 為什麼要測 CUDA RNG？

**短答（20–40 秒）**：GPU operations 或未來 stochastic CUDA behavior 可能消耗 CUDA RNG；如果 checkpoint 沒保存它，resume 後可能靜默改變 trajectory。即使目前 formal target dropout=0，完整 RNG state 仍是可重播 contract 的一部分。

**深入追問**：CPU RNG、CUDA RNG、data epoch seed 各自負責不同 stochastic domain，不能只保存一個 global seed 值。

**證據**：checkpoint schema；Phase 3/4 RNG restore and fresh-process replay tests；formal checkpoint audit。

### Q38. 為什麼要存 checkpoint SHA？

**短答（20–40 秒）**：SHA-256 將結果檔與具體 checkpoint bytes 綁定，可偵測替換、損毀或 median 計算後被修改。五個 final results 都記錄對應 final checkpoint hash。

**深入追問**：hash 證明 byte identity，不證明模型在科學上正確；因此還要做 schema、tensor finite、metadata、result sample count 與 log consistency audit。

**證據**：five hash manifests and `final_result.json`；`docs/formal_median_transcript.json`。

### Q39. 為什麼不用 best-test checkpoint？

**短答（20–40 秒）**：因為用 test set 選最佳 epoch 會把 test performance 回饋進 model selection，也會增加 post-hoc selection。formal rule 在訓練前固定為 epoch-200 final checkpoint only。

**深入追問**：若論文未唯一說明 checkpoint selection，本專案把它標為 explicit human-approved assumption，而不是假裝 paper 明文要求。

**證據**：`docs/assumptions.md`；`docs/reproduction_spec.md` checkpoint selection；five manifests。

## N. Determinism

### Q40. 這裡的 deterministic 是什麼意思？

**短答（20–40 秒）**：在記錄的同一 modern environment、相同 seed 與 policy 下，fresh processes 可重現相同短程 fingerprint；runtime 使用 FP32 eager、deterministic algorithms、cuDNN deterministic，並停用 AMP、TF32、compile 與 cuDNN benchmark。

**深入追問**：determinism 是有範圍的工程 property，不是跨不同 GPU、driver、PyTorch version 的永遠 bitwise guarantee。

**證據**：`docs/phase4_target_hardware_validation.md`；Phase 4 fresh-process probe；formal environment snapshots。

### Q41. Deterministic 是否保證和 paper 完全相同？

**短答（20–40 秒）**：不保證。它只控制本專案環境內的 replay；paper 的原始 seeds、完整 runtime、kernel versions 與某些細節未知，因此 deterministic modern run 仍可能和 paper trajectory 不同。

**深入追問**：determinism 解決的是「同 protocol/同環境能否穩定重做」，不是「不同年代的軟硬體是否產生同一浮點軌跡」。

**證據**：`docs/open_questions.md`；final report limitations；Phase 4 validation scope。

## O. Limitations

### Q42. 這個 reproduction 最大限制是什麼？

**短答（20–40 秒）**：無法取得 paper-era Table 5 runs 的完整 executable environment 與 seed identities，因此不能做 runtime-identical、trajectory-identical replication。它是證據驅動、假設預登記的 modern-framework reproduction。

**深入追問**：另一限制是 paper 只提供 aggregate median，沒有 individual runs，使 formal statistical equivalence analysis不可直接進行。

**證據**：`docs/open_questions.md`；`docs/final_reproduction_report.md` limitations。

### Q43. 4.40% 會不會代表 implementation 有錯？

**短答（20–40 秒）**：差異本身不能證明有錯，也不能證明沒錯。implementation 的可信度來自 architecture/optimizer/LR/resume/CUDA tests與 artifacts audit；performance 只做數值比較。要指認某個 bug 必須有可重現反例或 evidence mismatch。

**深入追問**：若發現 frozen semantics 實作與證據不符，該 formal result 就會被標 invalid，而不是用「接近 4.27」掩蓋。但目前 final audit 是 PASS。

**證據**：phase validation docs；final evidence audit；claim matrix。

## P. Future work / DenseNet

### Q44. 和一般 ResNet reproduction 相比，最大的學習是什麼？

**短答（20–40 秒）**：模型名稱或 parameter count 對上遠遠不夠。真正困難的是把 paper claim、official code、歷史 framework defaults與 modern assumptions 分開，並把 optimizer、data RNG、checkpoint與 aggregation都變成可測試 contract。

**深入追問**：WRN 特別顯示 width/depth 定義、pre-activation routing與 training micro-semantics會跨框架漂移；因此 evidence provenance 和 freeze process 本身就是研究成果。

**證據**：`docs/evidence_table.md`；phase 0–5 validation history；final report lessons learned。

### Q45. 為什麼 DenseNet 是合理的下一篇？

**短答（20–40 秒）**：DenseNet 延續「feature reuse 與 network connectivity」的研究脈絡，適合檢驗同一套 evidence-first 方法能否處理更複雜的 concatenation、growth rate、transition layer與記憶體語義。

**深入追問**：這只是 future-work rationale，不代表本 WRN result 能推導 DenseNet 的結果。下一專案仍需重新做 paper version、official code、dependency與 assumption freeze。

**證據**：`docs/final_reproduction_report.md` next work；本項屬 project methodological inference。

## 教授追殺題 / Adversarial defense questions

### A1. 「Torch7 exact revision 都不確定，你憑什麼叫 reproduction？」

**短答**：我不稱它為 paper-era bitwise replication，而稱為具 pre-registered modern-framework assumptions 的 protocol-faithful reproduction。所有不能唯一還原的地方都標 UNKNOWN 或 assumption，沒有藏在「照論文做」四個字裡。

**深入追問**：reproduction 的可信度來自 transparent scope、pinned evidence、freeze-before-results 和 validation；若要求 executable archaeology，應是另一個研究目標。

**證據**：`docs/reproduction_spec.md` scope；`docs/assumptions.md`；`docs/open_questions.md`。

### A2. 「現代框架不同，這不就只是你自己的 model？」

**短答**：framework 是 implementation choice，但 architecture/training decisions不是任意選擇；它們逐項追溯 paper、author code和historical dependencies，未決部分經人工核准並凍結。結果也未因接近或遠離 4.27% 而修改。

**深入追問**：若我把 PyTorch defaults直接當 paper fact，才會只是任意 reimplementation；本專案的 claim matrix 明確限制說法。

**證據**：`docs/evidence_table.md`；`docs/final_claim_matrix.md`；Git freeze history。

### A3. 「我為什麼要相信 4.40%？」

**短答**：它不是手抄的 summary；可由五個獨立 `final_result.json` 重算，並綁定 result/checkpoint hashes。最終 audit 驗證 45/45 artifacts、20/20 checkpoints、1,035/1,035 log events，median transcript payload SHA 也重新計算一致。

**深入追問**：每個 result 的 correct+incorrect 都是 10,000，且和 accuracy/error一致；五個 runs 均是同 baseline、200 epochs、78,000 updates、0 resume。

**證據**：formal run artifacts；`docs/formal_median_transcript.json`；final report result table。

### A4. 「0.13 pp 不就是你的 implementation 錯了嗎？」

**短答**：單一 aggregate difference 不能定位 implementation bug。implementation correctness 由語義測試與 artifact audit支持；0.13 pp 的可能來源很多，但沒有 controlled ablation就不能歸因。

**深入追問**：我保留「可能仍有未發現差異」作限制，但不以 performance proximity 取代 tests，也不以 tests 宣稱 paper-identical。

**證據**：phase validation documents；final report difference analysis and limitations。

### A5. 「既然先知道 4.27%，你的 freeze 仍然有 bias 吧？」

**短答**：知道 target result 是 reproduction 的必要條件，bias control 的重點是不能依本次 run outcomes反覆改 protocol。專案先建立 evidence hierarchy與人工核准 assumptions，再凍結 code/config，之後五個 seeds不變執行。

**深入追問**：完全 blinded 不一定可行，但完整 Git history、SHA與no-post-hoc rule讓 researcher degrees of freedom 可見並受限。

**證據**：Phase 0 freeze commit；formal baseline commit；five manifests。

### A6. 「seeds 1–5 有什麼科學意義？不就是隨便選？」

**短答**：具體數字不是 paper evidence，所以被標為 human-approved assumption；科學意義來自事先固定、全數執行、不可依結果排除，並維持 paper 的 five-run aggregation 結構。

**深入追問**：它們不代表 paper seeds或所有 RNG space。若改用另一個 seed set，應視為新預註冊 study。

**證據**：`docs/assumptions.md`；frozen config；formal authorization/run records。

### A7. 「median 為什麼有效？它可能藏掉壞 seed。」

**短答**：median 是 paper 明示與本專案事前凍結的 primary statistic；但我沒有隱藏 individual runs，五個 errors全部報告。這同時保留 paper comparability與資料透明度。

**深入追問**：median 對極端值較 robust，但五個 runs不足以做完整 distribution inference，所以不過度詮釋。

**證據**：paper Table 5 caption；five-run table；median transcript。

### A8. 「deterministic 了，為什麼不同 seeds 還會不同？」

**短答**：deterministic 的條件包含 seed；它保證同 seed在同環境重播相同 trajectory，不是讓不同 seeds 共用同一 initialization、data order與 augmentation。因此 between-seed variance仍然存在。

**深入追問**：這正是 paper採多 runs和median的原因之一；determinism降低不可控變異，不消除設計中的 stochastic factor。

**證據**：Phase 4 replay validation；frozen seed policy；formal results。

### A9. 「把 BN beta 做 weight decay 很奇怪，為什麼不改？」

**短答**：因為 frozen historical semantics 是 all trainable parameters 的 coupled decay。現代 recipe常排除 BN/bias，但那是不同 protocol；看到結果後改 scope會是 post-hoc tuning。

**深入追問**：專案以手算 optimizer tests驗證實作，不代表宣稱這是今日最佳 practice。

**證據**：evidence table weight-decay scope；Phase 3 optimizer tests；frozen config。

### A10. 「你如何證明 checkpoint resume 沒改 training？」

**短答**：把同一短程 run一邊不中斷、一邊在 mid-epoch存檔後 fresh-process恢復，最後比較 model、optimizer、momentum、BN、RNG、cursor與 fingerprint exact equality。

**深入追問**：formal runs本身0 resume，所以 formal result不依賴 resume；這項測試證明 engine 的 checkpoint contract，而 final audit又檢查20個 checkpoints schema。

**證據**：Phase 3/4 resume/replay tests；final checkpoint audit。

### A11. 「你選 epoch 200，是不是因為剛好最好？」

**短答**：不是。epoch-200 final-only rule在 formal training前凍結，沒有依 test curve挑 checkpoint。這也避免 test-set model selection。

**深入追問**：若 epoch 180更好也不能替換 primary result；任何 exploratory best-epoch analysis都必須另列且不能改 frozen aggregate。

**證據**：frozen spec/config；assumptions；five manifests。

### A12. 「45/45 hash 都對，只能證明檔案沒變，怎能說 technical valid？」

**短答**：同意 hash只證明 byte identity，所以 technical validity不是只靠 hash；還包括 architecture、optimizer、LR、gradient、resume、fresh-process CUDA tests，以及 checkpoint tensor/metadata和log consistency audit。

**深入追問**：證據是分層的：hash防篡改，schema/semantic checks驗內部一致，phase tests驗演算法contract，paper comparison則是另一層，不能互相取代。

**證據**：phase validation docs；final audit；`docs/final_reproduction_report.md` reproducibility section。

## 一句話防守底線

> 我重現的是一個事前凍結、來源分級、可稽核的現代框架 protocol；正式五-run median 是 4.40%，論文是 4.27%，差 +0.13 pp。沒有事前 tolerance，所以我報告差異，不事後把它包裝成 performance pass、failure 或 statistical equivalence。
