# Experiment Summary: PHASE3A_BASELINE

- **Experiment Type:** `baseline`
- **PSD Enabled:** `False` (Lambda PSD: `0.0`)
- **Steps Run:** `150`
- **Elapsed Time:** `1899.53s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `2678.72 MB`
- **Peak VRAM Reserved:** `5594.0 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8706` $\rightarrow$ **Final Training LM Loss:** `1.886`
- **Initial Validation LM Loss:** `3.6352` $\rightarrow$ **Final Validation LM Loss:** `1.3667`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `7.25`
- **ROUGE-1:** `50.72`
- **ROUGE-2:** `37.27`
- **ROUGE-L:** `46.42`
- **BERTScore Precision:** `87.93`
- **BERTScore Recall:** `91.87`
- **BERTScore F1:** `89.76`
- **Structured Content Coverage:** `100.0%`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Source Input:** `[PAGE] List of Governors of South Carolina [SECTION] Governors under the Constitution of 1868 [HEADE...`
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** [CELL] John C. Calhoun | [CELL] December 1, 1874 | [CELL] December 1, 1876 | [CELL] John C. Calhoun | [CELL] Republican | [CELL]  [ROW 4] [HCELL] 77 | [CELL] - | [CELL] John C. Calhoun | [CELL] December 1, 1876 | [CELL] December 1, 1878 | [CELL] John C. Calhoun | [CELL] Republican | [CELL]
- **Coverage:** 100.0%

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Source Input:** `[PAGE] Alma Jodorowsky [SECTION] Filmography [HEADER] Year | Title | Role | Notes [ROW 0] [HEADER] Y...`
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** 2014 | [CELL] The Last Days of the World | [CELL] Mireille | [CELL]  [ROW 8] [CELL] 2015 | [CELL] The Last Days of the World | [CELL] Mireille | [CELL]  [ROW 9] [CELL] 2016 | [CELL] The Last Days of the World | [CELL] Mireille | [CELL]  [ROW 10] [CELL] 2017 | [CELL] The Last Days of the World | [CELL] Mireille |
- **Coverage:** 100.0%

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Source Input:** `[PAGE] A. J. Hawk [SECTION] Career statistics [HEADER] Year | Team | GP | TOTAL | COMB | AST | SACK ...`
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** [CELL] 1 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL] 0 | [CELL
- **Coverage:** 100.0%

### Example 4 (`totto_-6915287003153277224` - `totto`)
- **Source Input:** `[PAGE] List of rulers of Brittany [SECTION] House of Montfort [HEADER] Name | Portrait | Birth | Mar...`
- **Target:** At the very end of his life, Arthur III became duke of Brittany, succeeding Peter II.
- **Prediction:** ] - | [CELL] 1401 son of John IV and Joan of Navarre | [CELL] (1) Joan of Navarre 1421 (2) Joan of Navarre 1422 (3) Joan of Navarre 1423 (4) Joan of Navarre 1424 (5) Joan of Navarre 1425 (6) Joan of Navarre 1426 (7) Joan of Navarre 1427 (8) Joan of Navarre 142
- **Coverage:** 100.0%

### Example 5 (`totto_-3004901021745997743` - `totto`)
- **Source Input:** `[PAGE] List of Speakers of the Minnesota House of Representatives [SECTION] State [HEADER] # | Speak...`
- **Target:** Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives.
- **Prediction:** [CELL] Bradley was a member of the Minnesota House of Representatives from 1857 to 1860. | [ROW 3] [CELL] 3 | [CELL] John S. Watrous | [CELL] 12 March 1859 | [CELL] 12 March 1860 | [CELL] Republican | [CELL] Watrous was the 3rd Speaker of the Minnesota House of Representatives. | [ROW 4] [CELL] 4 | [CELL] John S. Watrous | [CELL] 12 March 186
- **Coverage:** 100.0%
