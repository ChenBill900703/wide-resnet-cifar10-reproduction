# Project rules

本 repository 的目標是以可稽核證據重現 Wide Residual Networks 的 CIFAR-10 實驗。以下規則適用於 repository 全部目錄，除非日後更深層的 `AGENTS.md` 提供更嚴格且不衝突的規則。

# Evidence hierarchy

所有可影響正式 reproduction 的數字、架構、訓練設定與結果，都必須有可追溯證據。每筆證據都要記錄來源版本與足以重新定位的 locator。

證據優先級：

1. `PAPER-SPECIFIED`
   - `references/wide_residual_networks.pdf`
   - 論文正文、表格、圖、附錄
   - 必須記錄 page / section / table / figure。
   - arXiv 與 BMVC final 必須視為不同版本；不得默認內容相同。

2. `OFFICIAL-CODE-SPECIFIED`
   - 作者官方 repository：<https://github.com/szagoruyko/wide-residual-networks>
   - 必須記錄具體 file path、commit/hash（若可取得）、line/function/module。
   - 官方 code 不可冒充 paper specification；即使與 paper 一致，分類仍須分開記錄。

`DERIVED` 不是獨立來源層級。只有當輸入值皆已由上述可追溯來源支持、計算式完整記錄且無額外假設時，才可使用；應盡量寫成 `DERIVED FROM PAPER-SPECIFIED` 或 `DERIVED FROM OFFICIAL-CODE-SPECIFIED`。Derived 值不得冒充直接來源證據，也不得用來填補來源未知的語義。

3. `IMPLEMENTATION-ASSUMPTION`
   - 只有 paper 與 official code 都不能唯一決定時才允許提出。
   - 必須明確記錄理由、可能影響、替代方案與批准狀態。
   - 正式 training 前必須取得人工明示批准。

4. `UNKNOWN`
   - 查不到就標 `UNKNOWN`。
   - 禁止根據常見做法、模型記憶、其他 repository 或直覺補值。

## Official implementation lineage

作者 repository 內的 official implementation 必須依時間與用途分流：

- Paper-era / reference Torch7 implementation：優先用於還原 paper-era CIFAR training semantics。
- Later official PyTorch implementation：仍屬 `OFFICIAL-CODE-SPECIFIED`，但必須標記 subtype `later / auxiliary official PyTorch implementation`。它可用於 architecture cross-check；除非有可追溯證據證明它就是 paper-era training implementation，否則不得自動覆寫 Torch7 的 training semantics、initialization、batch-normalization defaults、dropout semantics、optimizer behavior、RNG behavior或其他 paper-era 行為。

若 Torch7 與 official PyTorch implementation 不一致，必須兩邊分列、記錄 `OFFICIAL-CODE-CONFLICT`，不得自行融合或選一邊，並交人工審核。

# Anti-hallucination rules

- 禁止依模型記憶填入論文數值。
- 禁止因為「通常 WRN 都這樣做」而當成論文設定。
- 禁止引用非作者第三方 PyTorch repository 作為 paper evidence。
- 第三方 implementation 可以作為 sanity check，但只能標為 `EXTERNAL-CROSS-CHECK`，且不得決定 frozen reproduction specification。
- 每一個重要數值都必須可以回答：「這個數字從哪一頁、哪一表、哪個官方 code 檔案來？」
- 如果 paper 與 official code 不一致，禁止自行融合；必須兩邊都記錄，明確標示 `CONFLICT`，交由人工決定。
- 如果 arXiv 與 BMVC final version 有差異，必須記錄版本、位置、值與差異，禁止挑選較有利數字而不揭露版本。
- 不得把 inference 當成 fact。推導值必須標示為推導，列出前提，且不得取代直接證據。
- 不得把 RTX 3070 Ti 或其他本地硬體的適配建議冒充論文設定；硬體適配只能是待批准的 `IMPLEMENTATION-ASSUMPTION`。
- 同一個結果數字必須連同 preprocessing、augmentation、dropout、run aggregation、batch size 與論文版本保存；禁止脫離上下文比較。

# Research logging

研究證據統一記錄在 `docs/evidence_table.md`，至少包含以下欄位：

| Item | Value | Classification | Paper location | Official-code location | Confidence | Conflict | Notes |
|---|---|---|---|---|---|---|---|

下列項目每一個都必須至少有一列；若來源或版本有多個值，必須分列：

- architecture depth formula
- widening factor
- stage widths
- block count
- block ordering
- dropout
- shortcut
- initialization
- preprocessing
- augmentation
- optimizer
- batch size
- momentum
- weight decay
- LR
- LR schedule
- epochs / iterations
- evaluation protocol
- CIFAR-10 reported errors
- parameter counts
- candidate reproduction configurations

新增或修改正式規格前，必須先更新 evidence table。任何尚未解決的選擇同時記入 `docs/open_questions.md`；任何擬採用但未由來源唯一決定的行為同時記入 `docs/assumptions.md`。

# Source acquisition / verification

允許並要求查資料查好查滿，但必須依下列來源分級：

Primary：

- BMVC 2016 official paper
- arXiv 1605.07146（保留版本號）
- authors' official GitHub repository

Secondary：

- 論文作者本人網站 / project page
- conference supplemental material

External cross-check only：

- torchvision
- timm
- third-party GitHub
- blogs
- paperswithcode
- StackOverflow
- discussions / issues

第三級來源不得決定 frozen reproduction specification。網頁來源應保存 URL、存取日期；GitHub 證據應優先使用 commit-pinned URL。PDF 必須以視覺檢查表格／圖與文字抽取交叉核對，不能只依賴搜尋摘要或 OCR。

# Conflict handling

- `PAPER-SPECIFIED` 與 `OFFICIAL-CODE-SPECIFIED` 的差異、BMVC 與 arXiv 的差異、以及同一來源內部不一致，都必須標記 `CONFLICT` 或 `CONFLICT/VARIANT`。
- 衝突未由人工決定前，相關欄位不得進入 frozen config。
- 版本新增資訊但沒有相反值時，記為 version extension；仍不得倒推早期版本採用了新增值。
- 可由多項直接證據算出的值只能標示為 derived cross-check，並保留計算式與全部前提。

# Professor-defense check

任何進入 frozen specification 的重要設定，都必須能回答：

1. 這個值是什麼？
2. 為什麼採用這個值？
3. 它是 `PAPER-SPECIFIED`、`OFFICIAL-CODE-SPECIFIED`、`DERIVED`、`IMPLEMENTATION-ASSUMPTION`，還是 `UNKNOWN`？
4. 精確來源位置在哪裡？
5. 是否存在不同 paper version、official implementation 或 source variant？
6. 若存在，為什麼本 project 選目前這一個？

任一問題無法回答時，不得標為 `CONFIRMED`，不得進入 frozen config。

# Target freeze rule

Formal reproduction target 必須同時明確綁定：

- paper version
- exact WRN configuration
- depth
- widening factor `k`
- preprocessing family
- augmentation
- dropout setting
- shortcut semantics
- training recipe
- evaluation protocol
- checkpoint selection rule
- run count
- seed policy
- aggregation method
- expected reference result

禁止只寫「WRN-16-8」、「WRN-28-10」或其他 model name 就視為完整 formal target。正式 target 必須由人工明示批准後，才允許解除 no-code gate。

# Phase lifecycle and freeze controls

Phase 0 已由人工明示批准結案。Frozen v1 formal target 為：

- Target name: `WRN16-8-CIFAR10-ARXIV-V4-MEANSTD-NODROPOUT-V1`
- Frozen specification: `docs/reproduction_spec.md`
- Frozen config: `configs/wrn16_8_arxiv_v4_frozen.yaml`
- Paper/result family: arXiv:1605.07146v4, Table 5
- Dataset/model: CIFAR-10, WRN-16-8, `B(3,3)`, mean/std, dropout 0
- Reference result: 4.27% error, median over 5 runs; paper parameter count 11.0M rounded

任何 frozen value 的修改都需要新的人工明示批准、版本化 config、evidence/assumption 更新與獨立 freeze commit。禁止為了改善結果而事後修改 frozen values。

## Active Phase 1 gate

目前只允許 architecture implementation + validation：

- PyTorch WRN model implementation
- architecture helpers與已批准的architecture-specific initialization
- unit tests與synthetic tensor tests
- parameter-count audit
- `docs/phase1_architecture_validation.md`

Phase 1 嚴格禁止：

- CIFAR-10或任何dataset download
- Dataset/DataLoader/sampler implementation
- augmentation implementation（只可保留未來Phase 2測試要求）
- optimizer或training loop implementation
- formal或smoke training
- hyperparameter tuning
- `runs/`、`checkpoints/`或training artifacts
- 自行開始Phase 2

Phase 1只有在所有architecture tests、synthetic forward/backward、parameter audit、`compileall`與diff checks通過後才可獨立commit。完成後必須停止等待人工Phase 1 review。
