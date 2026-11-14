import os
import glob
import json
from collections import defaultdict

def run():
    models = ['gemma_4b', 'llama_3b', 'qwen_3b', 'llama_8b', 'mistral_7b', 'phi_3_5', 'qwen_7b']
    print(f"{'Experiment':<28} | {'ToTTo (Prec / Halluc)':<24} | {'DART (Prec / Halluc)':<24} | {'WebNLG (Prec / Halluc)':<24}")
    print('-' * 105)

    for m in models:
        for mode in ['baseline', 'psd']:
            exp_dir = f'reports/experiments/long_context/{m}_{mode}_L1024'
            pred_file = os.path.join(exp_dir, 'sample_predictions.json')
            if not os.path.exists(pred_file):
                continue
            with open(pred_file, 'r', encoding='utf-8') as f:
                preds = json.load(f)
            by_ds = defaultdict(lambda: {'prec': [], 'halluc': []})
            for p in preds:
                ds = p.get('dataset', 'unknown')
                by_ds[ds]['prec'].append(p.get('factual_precision', 0.0))
                by_ds[ds]['halluc'].append(p.get('hallucination_rate', 0.0))
            
            t_p = sum(by_ds['totto']['prec'])/len(by_ds['totto']['prec']) if by_ds['totto']['prec'] else 0.0
            t_h = sum(by_ds['totto']['halluc'])/len(by_ds['totto']['halluc']) if by_ds['totto']['halluc'] else 0.0
            
            d_p = sum(by_ds['dart']['prec'])/len(by_ds['dart']['prec']) if by_ds['dart']['prec'] else 0.0
            d_h = sum(by_ds['dart']['halluc'])/len(by_ds['dart']['halluc']) if by_ds['dart']['halluc'] else 0.0
            
            w_p = sum(by_ds['webnlg']['prec'])/len(by_ds['webnlg']['prec']) if by_ds['webnlg']['prec'] else 0.0
            w_h = sum(by_ds['webnlg']['halluc'])/len(by_ds['webnlg']['halluc']) if by_ds['webnlg']['halluc'] else 0.0
            
            print(f"{m + '_' + mode:<28} | {t_p:5.1f}% / {t_h:5.1f}%          | {d_p:5.1f}% / {d_h:5.1f}%          | {w_p:5.1f}% / {w_h:5.1f}%")

if __name__ == '__main__':
    run()
