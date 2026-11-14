# Experiment Summary: PHASE4B_PSD_512

- **Experiment Type:** `psd`
- **PSD Enabled:** `True` (Lambda PSD: `0.1`)
- **Steps Run:** `500`
- **Elapsed Time:** `23998.22s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `4615.05 MB`
- **Peak VRAM Reserved:** `10598.0 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8259` $\rightarrow$ **Final Training LM Loss:** `1.1273`
- **Initial Validation LM Loss:** `3.265` $\rightarrow$ **Final Validation LM Loss:** `0.6917`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `22.3`
- **ROUGE-1:** `51.15`
- **ROUGE-2:** `33.99`
- **ROUGE-L:** `48.45`
- **BERTScore Precision:** `0.0`
- **BERTScore Recall:** `0.0`
- **BERTScore F1:** `0.0`
- **Structured Content Coverage:** `100.0%`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Source Input:** `[PAGE] List of Governors of South Carolina [SECTION] Governors under the Constitution of 1868 [HEADE...`
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** 
- **Coverage:** 100.0%

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Source Input:** `[PAGE] Alma Jodorowsky [SECTION] Filmography [HEADER] Year | Title | Role | Notes [ROW 0] [HEADER] Y...`
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** In 2016, Alma Jodorowsky starred in the film Kids in Love as Evelyn.
- **Coverage:** 100.0%

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Source Input:** `[PAGE] A. J. Hawk [SECTION] Career statistics [HEADER] Year | Team | GP | TOTAL | COMB | AST | SACK ...`
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** 
- **Coverage:** 100.0%

### Example 4 (`totto_-6915287003153277224` - `totto`)
- **Source Input:** `[PAGE] List of rulers of Brittany [SECTION] House of Montfort [HEADER] Name | Portrait | Birth | Mar...`
- **Target:** At the very end of his life, Arthur III became duke of Brittany, succeeding Peter II.
- **Prediction:** ) was the 12th ruler of the House of Montfort.
- **Coverage:** 100.0%

### Example 5 (`totto_-3004901021745997743` - `totto`)
- **Source Input:** `[PAGE] List of Speakers of the Minnesota House of Representatives [SECTION] State [HEADER] # | Speak...`
- **Target:** Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives.
- **Prediction:** 
- **Coverage:** 100.0%
