# Experiment Summary: PHASE4B_BASELINE_512

- **Experiment Type:** `baseline`
- **PSD Enabled:** `False` (Lambda PSD: `0.0`)
- **Steps Run:** `500`
- **Elapsed Time:** `8987.84s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `2752.92 MB`
- **Peak VRAM Reserved:** `8788.0 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8259` $\rightarrow$ **Final Training LM Loss:** `1.0671`
- **Initial Validation LM Loss:** `3.2661` $\rightarrow$ **Final Validation LM Loss:** `0.7082`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `7.92`
- **ROUGE-1:** `51.42`
- **ROUGE-2:** `33.69`
- **ROUGE-L:** `45.63`
- **BERTScore Precision:** `88.24`
- **BERTScore Recall:** `90.68`
- **BERTScore F1:** `89.34`
- **Structured Content Coverage:** `100.0%`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Source Input:** `[PAGE] List of Governors of South Carolina [SECTION] Governors under the Constitution of 1868 [HEADE...`
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** | [CELL] John D. Kennedy | [CELL] Democratic | [CELL]  [ROW 7] [HEADER] 80 | [CELL] - | [CELL] John D. Kennedy | [CELL] November 30, 1880 | [CELL] December 1, 1882 | [CELL] William Dunlap Simpson | [CELL] Democratic | [CELL]  [ROW 8] [HEADER] 81 | [CELL] - | [CELL] William Dunlap Simpson | [CELL] December 1, 1882 | [CELL]
- **Coverage:** 100.0%

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Source Input:** `[PAGE] Alma Jodorowsky [SECTION] Filmography [HEADER] Year | Title | Role | Notes [ROW 0] [HEADER] Y...`
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** In 2016, Alma Jodorowsky played the role of Evelyn in the film Kids in Love.
- **Coverage:** 100.0%

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Source Input:** `[PAGE] A. J. Hawk [SECTION] Career statistics [HEADER] Year | Team | GP | TOTAL | COMB | AST | SACK ...`
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** | [CELL] 63 | [CELL] 23 | [CELL] 3.0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 [ROW 5] [CELL] 2010 | [CELL] GB | [CELL] 16 | [CELL] 103 | [CELL] 73 | [CELL] 25 | [CELL]
- **Coverage:** 100.0%

### Example 4 (`totto_-6915287003153277224` - `totto`)
- **Source Input:** `[PAGE] List of rulers of Brittany [SECTION] House of Montfort [HEADER] Name | Portrait | Birth | Mar...`
- **Target:** At the very end of his life, Arthur III became duke of Brittany, succeeding Peter II.
- **Prediction:** ) 1450–1465 | [CELL] - | [CELL] 1450–1465 | [ROW 5] [CELL] Francis II the Good (Frañz II) 1465–1488 | [CELL] - | [CELL] 1465–1488 | [ROW 6] [CELL] Francis III the Great (Frañz III) 1488–1515 | [CELL] - | [CELL] 1488–1515 | [
- **Coverage:** 100.0%

### Example 5 (`totto_-3004901021745997743` - `totto`)
- **Source Input:** `[PAGE] List of Speakers of the Minnesota House of Representatives [SECTION] State [HEADER] # | Speak...`
- **Target:** Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives.
- **Prediction:** th [ROW 7] [CELL] 6 | [CELL] John A. Smith | [CELL] 12 January 1864 | [CELL] 12 January 1865 | [CELL] Republican | [CELL]  | [CELL] 6th [ROW 8] [CELL] 7 | [CELL] John A. Smith | [CELL] 12 January 11 | [CELL] 12 January 1865 | [CELL] Republican | [CELL]  | [CELL] 7th [ROW 9] [CELL
- **Coverage:** 100.0%
