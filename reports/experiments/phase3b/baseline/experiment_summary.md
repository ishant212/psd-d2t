# Experiment Summary: PHASE3B_BASELINE

- **Experiment Type:** `baseline`
- **PSD Enabled:** `False` (Lambda PSD: `0.0`)
- **Steps Run:** `500`
- **Elapsed Time:** `6652.33s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `2678.72 MB`
- **Peak VRAM Reserved:** `5598.0 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8706` $\rightarrow$ **Final Training LM Loss:** `1.0526`
- **Initial Validation LM Loss:** `3.6352` $\rightarrow$ **Final Validation LM Loss:** `1.3228`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `6.29`
- **ROUGE-1:** `46.15`
- **ROUGE-2:** `34.11`
- **ROUGE-L:** `44.94`
- **BERTScore Precision:** `86.91`
- **BERTScore Recall:** `90.13`
- **BERTScore F1:** `88.39`
- **Structured Content Coverage:** `100.0%`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Source Input:** `[PAGE] List of Governors of South Carolina [SECTION] Governors under the Constitution of 1868 [HEADE...`
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** [CELL] William Henry Smith | [CELL] December 1, 1874 | [CELL] December 1, 1876 | [CELL] William Henry Smith | [CELL] Republican | [CELL]  [ROW 4] [HCELL] 77 | [CELL] - | [CELL] William Henry Smith | [CELL] December 1, 1876 | [CELL] December 1, 1878 | [CELL] William Henry Smith | [CELL] Republican | [CELL]  [ROW 5] [HCELL] 78
- **Coverage:** 100.0%

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Source Input:** `[PAGE] Alma Jodorowsky [SECTION] Filmography [HEADER] Year | Title | Role | Notes [ROW 0] [HEADER] Y...`
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** 2014 | [CELL] The Diving Bell and the Butterfly | [CELL] Mireille | [CELL]  [ROW 8] [CELL] 2015 | [CELL] The Diving Bell and the Butterfly | [CELL] Mireille | [CELL]  [ROW 9] [CELL] 2016 | [CELL] The Diving Bell and the Butterfly | [CELL] Mireille | [CELL]  [ROW 10] [CELL] 2017 | [CELL] The Diving Bell and the Butterfly | [CELL]
- **Coverage:** 100.0%

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Source Input:** `[PAGE] A. J. Hawk [SECTION] Career statistics [HEADER] Year | Team | GP | TOTAL | COMB | AST | SACK ...`
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL] 1 | [CELL
- **Coverage:** 100.0%

### Example 4 (`totto_-6915287003153277224` - `totto`)
- **Source Input:** `[PAGE] List of rulers of Brittany [SECTION] House of Montfort [HEADER] Name | Portrait | Birth | Mar...`
- **Target:** At the very end of his life, Arthur III became duke of Brittany, succeeding Peter II.
- **Prediction:** ] - | [CELL] 1401 | [CELL] (1) Joan of Navarre, daughter of John IV, 1423 (2) Joan of Navarre, daughter of John IV, 1425 (3) Joan of Navarre, daughter of John IV, 1427 | [CELL] 1442 [ROW 3] [CELL] John VI (Yann VI) 1442–1465 | [CELL] - | [CELL] 1443 | [CELL] (1) Joan
- **Coverage:** 100.0%

### Example 5 (`totto_-3004901021745997743` - `totto`)
- **Source Input:** `[PAGE] List of Speakers of the Minnesota House of Representatives [SECTION] State [HEADER] # | Speak...`
- **Target:** Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives.
- **Prediction:** [CELL] Bradley was a Republican. | [ROW 3] [CELL] 3 | [CELL] John S. Watrous | [CELL] 12 December 1859 | [CELL] 12 March 1860 | [CELL] Unknown | [CELL] Watrous was a Republican. | [ROW 4] [CELL] 4 | [CELL] John S. Watrous | [CELL] 12 March 1860 | [CELL] 12 December 1861 | [CELL] Unknown | [CELL] Watrous was a
- **Coverage:** 100.0%
