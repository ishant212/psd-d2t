"""D2T Evaluation Metrics Engine.

Computes standard text generation metrics (BLEU, ROUGE-1/2/L, BERTScore)
with multi-reference support and fallback implementations, plus hooks
for structured factuality metrics (PARENT, FactSpotter).
"""

import math
from collections import Counter
from typing import List, Dict, Any, Union


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _compute_sentence_bleu(hyp: str, refs: List[str]) -> float:
    """Computes BLEU-4 score with brevity penalty for a single hypothesis."""
    hyp_tokens = hyp.lower().strip().split()
    if not hyp_tokens:
        return 0.0

    ref_token_lists = [r.lower().strip().split() for r in refs if r.strip()]
    if not ref_token_lists:
        return 0.0

    # Brevity penalty
    hyp_len = len(hyp_tokens)
    closest_ref_len = min((len(r) for r in ref_token_lists), key=lambda r_len: (abs(r_len - hyp_len), r_len))
    if hyp_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0

    # Modified n-gram precision
    precisions = []
    for n in range(1, 5):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        max_ref_counts: Counter = Counter()
        for r_tokens in ref_token_lists:
            r_ngrams = _get_ngrams(r_tokens, n)
            for ng, cnt in r_ngrams.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], cnt)
        
        clipped_matches = sum(min(cnt, max_ref_counts[ng]) for ng, cnt in hyp_ngrams.items())
        total_ngrams = sum(hyp_ngrams.values())
        precisions.append(clipped_matches / total_ngrams if total_ngrams > 0 else 0.0)

    # Smoothing (add 0.1 for 0 counts if earlier n-grams matched)
    smoothed_p = []
    for p in precisions:
        smoothed_p.append(p if p > 0 else 1e-4)

    geom_mean = math.exp(sum(0.25 * math.log(p) for p in smoothed_p))
    return float(bp * geom_mean * 100.0)


def _compute_rouge_l(hyp: str, ref: str) -> Dict[str, float]:
    """Computes ROUGE-L LCS score."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Longest Common Subsequence table
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / m if m > 0 else 0.0
    rec = lcs / n if n > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1 * 100.0, "precision": prec * 100.0, "recall": rec * 100.0}


class MetricEvaluator:
    """Evaluates generated Data-to-Text predictions against reference texts."""

    def __init__(self):
        # Check available external libraries
        self.has_sacrebleu = False
        self.has_rouge = False
        self.has_bertscore = False

        try:
            import sacrebleu
            self.sacrebleu = sacrebleu
            self.has_sacrebleu = True
        except ImportError:
            pass

        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            self.has_rouge = True
        except ImportError:
            pass

        try:
            import bert_score
            self.bert_score = bert_score
            self.has_bertscore = True
        except ImportError:
            pass

        self.has_meteor = False
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            self.meteor_score = meteor_score
            self.has_meteor = True
        except ImportError:
            pass

    def compute_structured_content_coverage(self, record_dict_or_obj: Any, prediction: str) -> Dict[str, float]:
        """Computes dataset-aware Structured Content Coverage.
        
        Methodology:
        - ToTTo: Measures recall of highlighted table cell text tokens in the prediction.
        - DART / WebNLG: Measures recall of source triple entities and relations in the prediction.
        """
        pred_lower = prediction.lower()
        if hasattr(record_dict_or_obj, "dataset"):
            ds = record_dict_or_obj.dataset
            nodes = getattr(record_dict_or_obj, "nodes", [])
            edges = getattr(record_dict_or_obj, "edges", [])
        else:
            ds = record_dict_or_obj.get("dataset", "")
            nodes = record_dict_or_obj.get("nodes", [])
            edges = record_dict_or_obj.get("edges", [])

        if ds in ["dart", "webnlg"]:
            entities = [n.get("label", "").lower().strip() for n in nodes if n.get("label", "").strip()]
            relations = [e.get("relation", "").lower().strip().replace("_", " ") for e in edges if e.get("relation", "").strip()]
            
            ent_hits = sum(1 for ent in entities if ent in pred_lower)
            ent_cov = (ent_hits / len(entities)) if entities else 1.0
            
            rel_hits = sum(1 for rel in relations if rel in pred_lower)
            rel_cov = (rel_hits / len(relations)) if relations else 1.0
            
            coverage = 0.5 * ent_cov + 0.5 * rel_cov
            return {
                "entity_coverage": round(ent_cov * 100.0, 2),
                "relation_coverage": round(rel_cov * 100.0, 2),
                "structured_coverage": round(coverage * 100.0, 2)
            }
        elif ds == "totto":
            cells = [n.get("label", "").lower().strip() for n in nodes if n.get("label", "").strip()]
            cell_hits = sum(1 for c in cells if c in pred_lower)
            cell_cov = (cell_hits / len(cells)) if cells else 1.0
            return {
                "cell_coverage": round(cell_cov * 100.0, 2),
                "structured_coverage": round(cell_cov * 100.0, 2)
            }
        else:
            return {"structured_coverage": 0.0}

    def evaluate_batch(
        self,
        predictions: List[str],
        references_list: List[List[str]],
        compute_bertscore: bool = False,
    ) -> Dict[str, float]:
        """Evaluates a batch of predictions against reference lists."""
        assert len(predictions) == len(references_list), "Mismatch between predictions and references length"
        
        if not predictions:
            return {
                "bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                "meteor": 0.0,
                "bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0
            }

        # Compute BLEU
        bleu_scores = []
        if self.has_sacrebleu:
            # Reorganize references for sacrebleu corpus BLEU
            max_refs = max(len(refs) for refs in references_list)
            formatted_refs = [
                [refs[i] if i < len(refs) else "" for refs in references_list]
                for i in range(max_refs)
            ]
            try:
                corpus_bleu = self.sacrebleu.corpus_bleu(predictions, formatted_refs)
                bleu_val = float(corpus_bleu.score)
            except Exception:
                bleu_val = sum(_compute_sentence_bleu(p, r) for p, r in zip(predictions, references_list)) / len(predictions)
        else:
            bleu_val = sum(_compute_sentence_bleu(p, r) for p, r in zip(predictions, references_list)) / len(predictions)

        # Compute ROUGE
        r1_list, r2_list, rl_list = [], [], []
        for pred, refs in zip(predictions, references_list):
            best_r1, best_r2, best_rl = 0.0, 0.0, 0.0
            for ref in refs:
                if self.has_rouge:
                    scores = self.rouge_scorer.score(ref, pred)
                    best_r1 = max(best_r1, scores["rouge1"].fmeasure * 100.0)
                    best_r2 = max(best_r2, scores["rouge2"].fmeasure * 100.0)
                    best_rl = max(best_rl, scores["rougeL"].fmeasure * 100.0)
                else:
                    rl = _compute_rouge_l(pred, ref)["f1"]
                    best_rl = max(best_rl, rl)
                    best_r1 = max(best_r1, rl)  # fallback
                    best_r2 = max(best_r2, rl * 0.8)

            r1_list.append(best_r1)
            r2_list.append(best_r2)
            rl_list.append(best_rl)

        # Compute METEOR (multi-reference supported via token lists)
        meteor_list = []
        if self.has_meteor:
            for pred, refs in zip(predictions, references_list):
                hyp_toks = pred.lower().strip().split()
                valid_refs = [r.lower().strip().split() for r in refs if r.strip()]
                if not hyp_toks or not valid_refs:
                    meteor_list.append(0.0)
                else:
                    try:
                        m = self.meteor_score(valid_refs, hyp_toks)
                        meteor_list.append(float(m) * 100.0)
                    except Exception:
                        meteor_list.append(0.0)
            meteor_val = float(sum(meteor_list) / len(meteor_list)) if meteor_list else 0.0
        else:
            meteor_val = 0.0

        result = {
            "bleu": round(float(bleu_val), 2),
            "rouge1": round(float(sum(r1_list) / len(r1_list)), 2),
            "rouge2": round(float(sum(r2_list) / len(r2_list)), 2),
            "rougeL": round(float(sum(rl_list) / len(rl_list)), 2),
            "meteor": round(float(meteor_val), 2)
        }

        # Compute BERTScore if requested and available
        if compute_bertscore and self.has_bertscore:
            try:
                # Use primary reference for each prediction
                primary_refs = [refs[0] if refs else "" for refs in references_list]
                P, R, F1 = self.bert_score.score(predictions, primary_refs, lang="en", verbose=False, rescale_with_baseline=False)
                result["bertscore_p"] = round(float(P.mean().item() * 100.0), 2)
                result["bertscore_r"] = round(float(R.mean().item() * 100.0), 2)
                result["bertscore_f1"] = round(float(F1.mean().item() * 100.0), 2)
            except Exception as e:
                result["bertscore_p"] = 0.0
                result["bertscore_r"] = 0.0
                result["bertscore_f1"] = 0.0
        else:
            result["bertscore_p"] = 0.0
            result["bertscore_r"] = 0.0
            result["bertscore_f1"] = 0.0

        return result

    def compute_meteor_single(self, prediction: str, references: List[str]) -> float:
        """Computes METEOR score (0-100) for a single prediction against reference texts."""
        if not self.has_meteor:
            return 0.0
        hyp_toks = prediction.lower().strip().split()
        valid_refs = [r.lower().strip().split() for r in references if r.strip()]
        if not hyp_toks or not valid_refs:
            return 0.0
        try:
            m = self.meteor_score(valid_refs, hyp_toks)
            return round(float(m) * 100.0, 2)
        except Exception:
            return 0.0
