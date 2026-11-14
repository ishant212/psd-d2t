# Experiment Summary: PSD_QWEN3_4B_LORA

- **Experiment Type:** `psd`
- **PSD Enabled:** `True` (Lambda PSD: `0.1`)
- **Steps Run:** `10`
- **Elapsed Time:** `342.74s`
- **Trainable Parameters:** `2,949,120` / `2,208,759,296` (`0.1335%`)
- **Peak VRAM Allocated:** `2689.72 MB`

## Loss Progression
- **Initial Training LM Loss:** `3.8706` $\rightarrow$ **Final Training LM Loss:** `3.2839`
- **Initial Validation LM Loss:** `3.6376` $\rightarrow$ **Final Validation LM Loss:** `2.1603`

## Generation Metrics (Validation Sample)
- **BLEU:** `0.2`
- **ROUGE-1:** `5.43`
- **ROUGE-2:** `0.0`
- **ROUGE-L:** `5.43`

## Sample Output Preview

### Example 1 (`totto_7391450717765563190` - `totto`)
- **Target:** Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.
- **Prediction:** [CELL] William A. B. Smith | [CELL] December 1, 1874 | [CELL] December 1, 11875 | [CELL] William A. B. Smith | [CELL] Republican | [CELL]  [ROW 4] [HCELL] 77 | [CELL] - | [CELL] William A. B. Smith | [CELL] December 1, 1875 | [CELL] December 1, 1876 | [CELL] William A. B. Smith | [CELL] Republican | [CELL]

### Example 2 (`totto_9012083751335522596` - `totto`)
- **Target:** Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.
- **Prediction:** 2014 | [CELL] The Last of the Moors | [CELL] Mireille | [CELL]  [ROW 8] [CELL] 2015 | [CELL] The Last of the Moors | [CELL] Mireille | [CELL]  [ROW 9] [CELL] 2016 | [CELL] The Last of the Moors | [CELL] Mireille | [CELL]  [ROW 10] [CELL] 2017 | [CELL] The Last of the Moors | [CELL] Mireille |

### Example 3 (`totto_-8764917516249435941` - `totto`)
- **Target:** In his rookie season, Hawk led with 119 total tackles.
- **Prediction:** [CELL] 0 [ROW 2] [HEADER] 2007 | [HEADER] 2007 | [HEADER] 16 | [HEADER] 122 | [HEADER] 83 | [HEADER] 38 | [HEADER] 3.5 | [HEADER] 1 | [HEADER] 2 | [HEADER] 0 | [HEADER] 2 | [HEADER] 31 | [HEADER] 14 | [HEADER] 1 | [HEADER] 0 | [HEADER] 0 [ROW 3] [HEADER]
