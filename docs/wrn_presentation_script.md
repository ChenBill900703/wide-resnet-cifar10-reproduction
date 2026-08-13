# Wide Residual Networks CIFAR-10 Reproduction — 完整簡報講稿

預計總長：**14 分 10 秒**（不含提問）
語言：繁體中文，technical terms 保留英文。
口說原則：先說證據範圍，再說數字；不把 modern-framework reproduction 說成 paper-era bitwise replication。

## Slide 1 — Wide Residual Networks CIFAR-10 Reproduction

**目標時間：35 秒**

**完整講稿**：

各位老師好，我的題目是 Wide Residual Networks 在 CIFAR-10 上的重現研究。我選定的正式 target 是 WRN-16-8、mean/std normalization、no dropout，對應 arXiv v4 Table 5。這份工作不只是把模型跑出一個 accuracy，而是建立從論文、作者程式碼、歷史 dependency，到 frozen config、五次正式執行與結果 hash 的完整證據鏈。最後的五-run median test error 是 4.40%，論文參考值是 4.27%。接下來我會先說明 WRN 的核心，再交代 protocol、結果與限制。

**不要過度宣稱**：不要在開場說「完全重現論文」或「證明論文正確」。

**轉場**：先從為什麼值得重現 WRN 開始。

**記住的關鍵數字**：4.40% vs 4.27%。

**可能被打斷的問題**：你所謂的「重現」範圍是什麼？

**安全短答**：是具預先登記 modern-framework assumptions 的 protocol-faithful reproduction，不是 Torch7 bitwise replication。

**Evidence footer**：Paper: arXiv:1605.07146v4, Table 5；Project baseline: `225cf8d...`。

## Slide 2 — 為什麼重現 WRN？

**目標時間：45 秒**

**完整講稿**：

WRN 的研究背景是 residual learning 已經讓很深的 network 可以訓練，但持續加深會增加序列計算成本，而且每一層可能相對窄。WRN 的核心問題是：如果把 capacity 從 depth 部分轉移到 width，能不能用較淺、較寬的 residual network 得到更好的效率與表現權衡。它很適合做 reproduction，因為表面上 model name 很簡單，但真正影響結果的 block routing、normalization、optimizer semantics、seed 和 aggregation 都分散在不同證據來源。也就是說，它可以測試我們能不能把一篇經典論文轉成可稽核的現代 protocol。

**不要過度宣稱**：不要說 width 在所有任務都比 depth 好。

**轉場**：下一頁把論文的 width-versus-depth 主張拆開。

**記住的關鍵數字**：WRN 論文年份 2016。

**可能被打斷的問題**：這和一般 ResNet 有什麼不同？

**安全短答**：保留 residual connections，但用 widening factor 重新配置每個 stage 的 capacity。

**Evidence footer**：Paper: abstract, §1–3。

## Slide 3 — 論文核心：不是只追求更深

**目標時間：50 秒**

**完整講稿**：

這一頁的重點不是把 WRN 說成「加很多 channel」而已。Residual block 的 shortcut 仍然保留，改變的是 residual branch 的寬度。論文以 widening factor `k` 表示這個擴張；對 CIFAR 架構，stage widths 是 16、16k、32k、64k。增加 width 能提高每個 block 的 feature capacity，也更容易利用 GPU 的平行計算。相對地，單純增加 depth 會拉長 sequential path。我要強調，這是論文在它的模型與實驗範圍內提出並驗證的設計主張，不是 width 永遠優於 depth 的一般定理。

**不要過度宣稱**：不要把 empirical argument 說成數學定理。

**轉場**：有了這個概念後，下一頁定義本專案究竟重現哪一個表格 cell。

**記住的關鍵數字**：stage family 16、16k、32k、64k。

**可能被打斷的問題**：為什麼 width 能加速？

**安全短答**：較寬的卷積有較多可平行工作；較深的網路有較長序列依賴，但實際速度仍取決於硬體與 kernel。

**Evidence footer**：Paper: p.4 Table 1；§3–4。

## Slide 4 — Formal target：先凍結，再看結果

**目標時間：55 秒**

**完整講稿**：

正式 target 完整名稱是 WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1。它綁定 arXiv v4、Table 5、CIFAR-10、WRN-16-8、mean/std preprocessing、dropout 0、五次執行的 median。論文參考 test error 是 4.27%，表中參數量是四捨五入的一位小數 11.0M。本實作精確 trainable parameter count 是 10,961,370，也就是 10.96137M，四捨五入後與 11.0M 一致。這個 target、config 和 SHA 都在正式訓練前凍結，避免看到結果後改問題。

**不要過度宣稱**：parameter count 對上不等於所有 training semantics 都完全相同。

**轉場**：下一頁看 10,961,370 個參數如何分布在 WRN-16-8 架構。

**記住的關鍵數字**：4.27%；10,961,370；YAML SHA 開頭 `18EF6815`。

**可能被打斷的問題**：為什麼不是精確 11,000,000？

**安全短答**：論文以一位小數 million 報告；10.96137M 在相同粒度四捨五入為 11.0M。

**Evidence footer**：Paper: p.8 Table 5；Frozen YAML SHA: `18EF6815...959B3`。

## Slide 5 — WRN-16-8 architecture

**目標時間：60 秒**

**完整講稿**：

`16-8` 的 16 是 nominal depth，8 是 widening factor。CIFAR WRN 使用 `n=6N+4`，所以 `N=(16-4)/6=2`，也就是三個 residual groups 各有兩個 `B(3,3)` blocks。32×32 input 先經 16-channel stem；Group 1 是 128 channels、維持 32×32；Group 2 是 256、降到 16×16；Group 3 是 512、降到 8×8。最後經 BN、ReLU、global average pooling 和 10-class classifier。dimension-changing shortcut 與 pre-activation routing 依 frozen official Torch7 evidence 實作和測試；我不把這些 micro-semantics 全部冒充成 Table 5 直接寫出的內容。

**不要過度宣稱**：不要簡化成每個 stage 只有「兩層」；是每 stage 兩個 `B(3,3)` blocks。

**轉場**：架構只是第一層，下一頁說明如何管理來源衝突與未知項。

**記住的關鍵數字**：N=2；channels 16→128→256→512。

**可能被打斷的問題**：為什麼 depth 公式會是 6N+4？

**安全短答**：三個 groups、每 block 兩個 convolution 形成 6N，再加 CIFAR WRN 的 stem與尾端計數慣例。

**Evidence footer**：Paper: p.4 Table 1, §3；Project: Phase 1 architecture validation。

## Slide 6 — Evidence governance：來源不能混在一起

**目標時間：45 秒**

**完整講稿**：

我把每個重要決策分為六類。論文直接寫的是 `PAPER-SPECIFIED`；作者 repository 是 `OFFICIAL-CODE-SPECIFIED`；Torch 當年 module defaults 屬於 `HISTORICAL-DEPENDENCY-BACKED`；可由已知輸入算出的數字標 `DERIVED`；來源不能唯一決定的現代 port 行為是 `IMPLEMENTATION-ASSUMPTION`；五次正式跑出的 4.40% 才是 `FORMAL-REPRODUCTION-RESULT`。這個分層的目的，是避免把 PyTorch default 說成論文設定，也避免用結果接近來倒推「一定做對」。

**不要過度宣稱**：不要說所有數字都來自論文。

**轉場**：下一頁用 data pipeline 示範，同一流程裡其實同時存在 paper-level 與 code-level evidence。

**記住的關鍵數字**：6 類 evidence classification。

**可能被打斷的問題**：來源衝突時怎麼辦？

**安全短答**：兩邊分列為 conflict，未經人工決定前不進 frozen config。

**Evidence footer**：Project: `docs/evidence_table.md`, `docs/assumptions.md`, `docs/open_questions.md`。

## Slide 7 — CIFAR-10 data pipeline

**目標時間：50 秒**

**完整講稿**：

CIFAR-10 是 50,000 train、10,000 test。輸入採 byte scale 0 到 255，channel mean 是 125.3、123.0、113.9，std 是 63.0、62.1、66.7。訓練 augmentation 的凍結順序是 horizontal flip、reflection pad 4、random 32×32 crop，最後 normalization；flip probability 0.5，crop offsets 每個方向是 0 到 8。mean/std family和 moderate augmentation有 paper evidence，但數值尺度、順序與 RNG mapping 屬 official-code或 project-level micro-semantics。每個 one-based epoch number作為 data seed，讓 permutation與augmentation可重播。

**不要過度宣稱**：不要把 transform 的每個邊界細節都稱為 paper-specified。

**轉場**：資料固定後，下一頁是 optimizer、schedule 與 exact update accounting。

**記住的關鍵數字**：50,000/10,000；mean 125.3/123.0/113.9。

**可能被打斷的問題**：為什麼不先除以 255？

**安全短答**：formal protocol凍結的是 author-code evidence支持的 byte-scale mean/std semantics；不是套用現代 torchvision default。

**Evidence footer**：Paper: p.6 §4.2, p.8 Table 5；Project: Phase 2 validation。

## Slide 8 — Training protocol 與 LR schedule

**目標時間：60 秒**

**完整講稿**：

Optimizer 是 SGD：initial LR 0.1、momentum 0.9、dampening 0、Nesterov true、coupled weight decay 0.0005，而且 scope 是全部 trainable parameters，包含 BN gamma、beta 和 FC bias。Loss 是 mean CrossEntropyLoss，label smoothing 0。Batch size 128、`drop_last=true`，所以每 epoch 是 `floor(50000/128)=390` updates、49,920 sample presentations；200 epochs 合計 78,000 updates。LR 在 one-based epoch 開始時切換：1到59是 .1、60到119是 .02、120到159是 .004、160到200是 .0008。投影片下方的 update ranges是從390推導的 cross-check，不是論文原句。

**不要過度宣稱**：不要把 390 與 78,000 說成 paper 直接報告。

**轉場**：有 protocol 還不夠，下一頁說明如何驗證它真的按照定義執行。

**記住的關鍵數字**：390 updates/epoch；78,000 total。

**可能被打斷的問題**：drop_last 少 80 張是否造成 bias？

**安全短答**：每 epoch少80個位置，但每次 permutation不同；不能聲稱零影響，所以將它預先凍結並揭露。

**Evidence footer**：Frozen config；Phase 3 optimizer/LR boundary validation。

## Slide 9 — Reproducibility engineering

**目標時間：55 秒**

**完整講稿**：

這個專案把「可重現」拆成可測試的 contract。Architecture有參數量、shape、synthetic forward/backward與initialization tests；optimizer用手算第一、第二步驗證 Nesterov和coupled decay；LR測boundary與global update。Checkpoint不是只存 weights，而是 model、optimizer、momentum、BN buffers、CPU/CUDA RNG與mid-epoch cursor。測試把 uninterrupted run和fresh-process resume做 exact equality比較。最後還在RTX 3070 Ti上做fresh-process deterministic fingerprint與CIFAR GPU preflight。這些證據支持技術有效性，但不等於跨2016與2026環境的bitwise identity。

**不要過度宣稱**：deterministic 不代表 paper-identical。

**轉場**：下一頁把正式硬體與 numerical policy 完整攤開。

**記住的關鍵數字**：20/20 checkpoints 最終通過 semantic audit。

**可能被打斷的問題**：為什麼 dropout=0 還要存 CUDA RNG？

**安全短答**：完整 RNG state是resume contract，GPU operation或未來 stochastic path都可能消耗它；只存單一seed不夠。

**Evidence footer**：Project: Phase 1–4 validation；checkpoint/replay tests。

## Slide 10 — RTX 3070 Ti target environment

**目標時間：45 秒**

**完整講稿**：

五個 formal runs使用同一個現代環境：Python 3.11.9、PyTorch 2.13.0加CUDA 12.6、torchvision 0.28.0、cuDNN 91002、driver 591.86，GPU是RTX 3070 Ti，compute capability 8.6。numerical policy是FP32 eager execution，AMP、TF32與torch.compile都關閉；cuDNN benchmark關閉，cuDNN deterministic和deterministic algorithms開啟。這些是本次environment-observed evidence，不是論文硬體設定，也不宣稱和原作者runtime相同。

**不要過度宣稱**：不要說這台GPU是論文使用或「最適合」WRN的GPU。

**轉場**：環境固定後，下一頁說正式五次執行如何避免挑seed與事後調參。

**記住的關鍵數字**：RTX 3070 Ti；compute capability 8.6。

**可能被打斷的問題**：8GB VRAM為什麼顯示略低於8×1024³？

**安全短答**：device API回報8,589,410,304 bytes；專案以實際device identity與可執行preflight為準，不把商品標示容量當硬性二進位閾值。

**Evidence footer**：Project: Phase 4 CUDA validation；five formal manifests。

## Slide 11 — Formal five-run governance

**目標時間：50 秒**

**完整講稿**：

正式 code baseline 是完整 commit `225cf8d44c36a8a210f6989bf76b9ebfe460adbd`，frozen YAML SHA開頭是 `18EF6815`。預先登記的 project seeds 是1到5；這五個不是宣稱為paper seeds，而是human-approved assumptions。每個run都必須完成200 epochs、78,000 optimizer updates、resume count 0，結果只取epoch 200 final checkpoint。五個seed逐一取得明示授權，全部完成後才做median；沒有依單一run結果改設定，也沒有排除表現較差的seed。

**不要過度宣稱**：不要說 seeds 1–5 就是作者原始 seeds。

**轉場**：下一頁直接看每個 seed 的實際結果，不只展示median。

**記住的關鍵數字**：5 runs；0 resume；baseline `225cf8d...`。

**可能被打斷的問題**：既然知道4.27%，freeze真的沒有bias嗎？

**安全短答**：target值本來就已知；bias control是freeze後不依本次outcome修改protocol，且所有決策與SHA可稽核。

**Evidence footer**：Formal baseline: `225cf8d...`；Frozen YAML SHA: `18EF6815...959B3`。

## Slide 12 — 五個正式 runs：完整結果

**目標時間：75 秒**

**完整講稿**：

這張表是從五個formal `final_result.json`逐一讀取，不是從排序值猜seed mapping。Seed 1是4.41%，Seed 2是4.40%，Seed 3是4.32%，Seed 4是4.28%，Seed 5是4.43%；每次都是10,000張test samples。排序後是4.28、4.32、4.40、4.41、4.43，所以中央值也就是frozen median是4.40%，對應Seed 2，正確9,560、錯誤440。圖上的paper line 4.27只作reference。完整性稽核另外確認45/45 formal artifact hashes、20/20 checkpoints和1,035/1,035 log events全部通過。這些數字證明結果可追溯，不是說hash通過就證明performance等價。

**不要過度宣稱**：不要把 Seed 4 的4.28%挑成正式結果；預註冊統計量是median。

**轉場**：下一頁把唯一正式comparison——4.27對4.40——公平地並列。

**記住的關鍵數字**：4.41、4.40、4.32、4.28、4.43；median 4.40。

**可能被打斷的問題**：為什麼不用最好seed 4.28？

**安全短答**：那是post-hoc best-run selection；formal rule在訓練前固定為five-run median。

**Evidence footer**：Formal results: seed_01–seed_05；median transcript SHA: `993C9057...C49D`。

## Slide 13 — Paper 4.27% vs reproduction 4.40%

**目標時間：65 秒**

**完整講稿**：

在相同報告粒度下，paper的five-run median test error是4.27%，reproduction的frozen median是4.40%，純數值差是加0.13 percentage points。這不是相對0.13%，而是percentage-point difference。圖表用從0開始的axis，避免把很小差異視覺放大。最重要的解讀是：專案沒有在執行前核准performance tolerance、equivalence margin或significance test，所以我不把加0.13 pp事後分類成成功、失敗、等價或不顯著。我能主張的是frozen protocol下的technical validity，以及透明的numerical comparison。

**不要過度宣稱**：不要說「只差0.13所以完全重現」或「0.13不顯著」。

**轉場**：下一頁把已知差異與可能原因分開，避免事後歸因。

**記住的關鍵數字**：+0.13 percentage points。

**可能被打斷的問題**：那到底算成功還失敗？

**安全短答**：technical audit是PASS；performance沒有事前tolerance，所以只報4.40對4.27，不事後判定pass/fail。

**Evidence footer**：Paper: Table 5；Formal median transcript；no pre-approved tolerance。

## Slide 14 — 為什麼仍可能有差異？

**目標時間：50 秒**

**完整講稿**：

我把差異分析分成兩層。已知差異包括：本專案是modern PyTorch port而不是原始Torch7 ecosystem；CUDA、cuDNN與hardware不同；project seeds不宣稱是paper seeds；部分micro-semantics必須由author code、historical dependencies與explicit assumptions凍結。可能的contributors則包括floating-point reduction順序、library kernels、未公開的原始runtime細節，以及只有五個runs時的stochastic distribution。這些因素都合理，但沒有做controlled ablation，所以不能說其中任何一項「造成」加0.13 pp。

**不要過度宣稱**：不要把可能因素說成已證實因果。

**轉場**：因此最後的成果應該分成「已完成的技術重現」與「仍保留的performance不確定性」。

**記住的關鍵數字**：0 個已證實的單一 causal explanation。

**可能被打斷的問題**：差異會不會就是implementation bug？

**安全短答**：不能由aggregate difference判定；semantic tests與audit支持實作，但仍保留unknown，沒有證據就不歸因。

**Evidence footer**：Project: final report difference analysis；open questions/assumptions。

## Slide 15 — 技術上完成了什麼？

**目標時間：55 秒**

**完整講稿**：

這個專案成功完成三件事。第一，把paper、official code、historical dependency與assumptions轉成一份可追溯的frozen specification。第二，把architecture、data、optimizer、LR、checkpoint、RNG、resume與CUDA determinism變成可執行的validation，而不是口頭保證。第三，在同一baseline下完成五個200-epoch formal runs，所有artifacts通過integrity audit，並依預登記rule得到4.40%的median。最保守且準確的結論是：這是一個technically valid、protocol-faithful modern-framework reproduction；它不等於paper-era bitwise replication，也不證明原論文正確。

**不要過度宣稱**：不要用「完全成功」省略scope。

**轉場**：最後一頁整理限制，以及這套方法如何延伸到下一篇研究。

**記住的關鍵數字**：5/5 VALID；45/45、20/20、1,035/1,035 PASS。

**可能被打斷的問題**：hash通過就叫technically valid嗎？

**安全短答**：不是；hash只防替換，technical validity還來自phase tests、semantic checkpoint checks與log consistency。

**Evidence footer**：Project: Phase 1–5 validation；final evidence audit PASS。

## Slide 16 — Limitations 與下一步

**目標時間：50 秒**

**完整講稿**：

最大限制是原始Table 5五次run的seed identities、完整Torch7 dependency lock、GPU與kernel environment並未公開，因此不能宣稱trajectory-identical。第二，paper只提供median，沒有individual runs，使我們不能直接做嚴格的statistical equivalence分析。第三，五個runs是為對齊paper aggregation，不足以完整估計performance distribution。下一步我會延用這套evidence-first方法研究DenseNet：重新做paper version與official code lineage、growth rate與transition semantics、dependency defaults、assumption approval與freeze，而不是把WRN結論直接外推。我的最後一句話是：正式median 4.40%，paper 4.27%，差加0.13 pp；證據完整、差異透明、不事後調參。

**不要過度宣稱**：不要說WRN結果保證DenseNet可重現。

**轉場**：謝謝老師，接下來我願意從paper evidence、optimizer semantics或formal artifacts任何一層接受提問。

**記住的關鍵數字**：4.40%、4.27%、+0.13 pp。

**可能被打斷的問題**：DenseNet為什麼是合理下一步？

**安全短答**：它延續feature reuse/connectivity議題，又增加growth rate、concatenation和transition等可稽核挑戰，適合驗證方法能否泛化。

**Evidence footer**：Project: final report limitations/next work；future work is methodological inference。

## 30 秒備用摘要

我重現的是 Zagoruyko 與 Komodakis 的 WRN 論文，正式 target 是 arXiv v4 Table 5 的 CIFAR-10 WRN-16-8、mean/std、no-dropout。專案先把 paper、official code、歷史 dependency 與 modern-framework assumptions 分類並凍結，再在同一 baseline 下完成 seeds 1–5 的五個 200-epoch runs。所有 artifacts 與 checkpoints 通過稽核；frozen median test error 是 4.40%，論文是 4.27%，差 +0.13 pp。因沒有事前 tolerance，我只做透明比較，不事後宣告 performance pass 或 failure。

## 2 分鐘備用摘要

這份研究的對象是 2016 年的 *Wide Residual Networks*。WRN 的核心不是單純讓 network 更深，而是保留 residual learning，同時用 widening factor 增加每個 stage 的 capacity。我選定 arXiv v4 Table 5 的 CIFAR-10 WRN-16-8、mean/std、no-dropout 作為 formal target；paper 報告 11.0M parameters與 five-run median test error 4.27%。

重現的困難在於，論文並沒有唯一寫出所有可執行細節。因此我建立 evidence hierarchy，把論文、作者 Torch7 code、歷史 dependency behavior、可推導數字與需要人工核准的 modern PyTorch assumptions分開。所有正式設定在訓練前進入 frozen YAML，SHA-256 是 `18EF6815...959B3`；formal code baseline是 `225cf8d...`。

實作的 WRN-16-8 有 10,961,370 trainable parameters，四捨五入後與 paper 的11.0M一致。Training使用CIFAR-10、batch 128、每epoch 390 updates、200 epochs共78,000 updates；SGD、Nesterov、coupled weight decay和LR boundaries都用analytical與boundary tests驗證。Checkpoint還包含model、optimizer、BN、CPU/CUDA RNG與mid-epoch cursor，並驗證uninterrupted和resumed trajectory exact equality。

最後在同一 RTX 3070 Ti environment下完成seeds 1到5。五個test errors是4.41、4.40、4.32、4.28、4.43%，所以預先登記的median是4.40%。最終audit確認45/45 artifacts、20/20 checkpoints、1,035/1,035 log events通過。相較paper 4.27%，差+0.13 percentage points。因為沒有事前定義tolerance或統計等價界線，我的結論是technically valid、protocol-faithful modern-framework reproduction，並透明報告數值差異；不是宣稱完全重現4.27%，也沒有為了靠近paper做事後調參。
