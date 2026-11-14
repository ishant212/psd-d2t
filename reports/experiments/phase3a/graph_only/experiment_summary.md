# Experiment Summary: PHASE3A_GRAPH_ONLY

- **Experiment Type:** `psd`
- **PSD Enabled:** `True` (Lambda PSD: `0.1`)
- **Steps Run:** `150`
- **Elapsed Time:** `1757.24s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `2690.96 MB`
- **Peak VRAM Reserved:** `5602.0 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8706` $\rightarrow$ **Final Training LM Loss:** `1.9065`
- **Initial Validation LM Loss:** `3.6366` $\rightarrow$ **Final Validation LM Loss:** `1.4673`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `17.21`
- **ROUGE-1:** `49.04`
- **ROUGE-2:** `32.66`
- **ROUGE-L:** `45.73`
- **BERTScore Precision:** `89.25`
- **BERTScore Recall:** `90.39`
- **BERTScore F1:** `89.79`
- **Structured Content Coverage:** `100.0%`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Source Input:** `[PAGE] List of Governors of South Carolina [SECTION] Governors under the Constitution of 1868 [HEADE...`
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** [CELL] James A. Williams | [CELL] November 1, 1874 | [CELL] John A. Williams | [CELL] Democratic | [CELL] Williams was elected in 1876.
- **Coverage:** 100.0%

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Source Input:** `[PAGE] Alma Jodorowsky [SECTION] Filmography [HEADER] Year | Title | Role | Notes [ROW 0] [HEADER] Y...`
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** 2014 | [CELL] The Last Days of the World | [CELL] Alma | [CELL] [ROW 8] [CELL] 2015 | [CELL] The Last Days of the World | [CELL] Alma | [CELL]
- **Coverage:** 100.0%

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Source Input:** `[PAGE] A. J. Hawk [SECTION] Career statistics [HEADER] Year | Team | GP | TOTAL | COMB | AST | SACK ...`
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** [CELL] 1 | [CELL] 1
- **Coverage:** 100.0%

### Example 4 (`totto_-6915287003153277224` - `totto`)
- **Source Input:** `[PAGE] List of rulers of Brittany [SECTION] House of Montfort [HEADER] Name | Portrait | Birth | Mar...`
- **Target:** At the very end of his life, Arthur III became duke of Brittany, succeeding Peter II.
- **Prediction:** ] 1421–1442 | [CELL] 1442 | [CELL] 1442
- **Coverage:** 100.0%

### Example 5 (`totto_-3004901021745997743` - `totto`)
- **Source Input:** `[PAGE] List of Speakers of the Minnesota House of Representatives [SECTION] State [HEADER] # | Speak...`
- **Target:** Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives.
- **Prediction:** [CELL] Bradley was a Republican.
- **Coverage:** 100.0%
