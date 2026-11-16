"""Score eval_set_raw.csv with every saved sexism checkpoint -> eval_set_scored.csv (raw file untouched).
Usage: python scripts/evaluation/score_eval_set.py [in.csv] [out.csv]
Columns added: p_<model> = P(sexist) in [0,1]. Stage1 = Civil Comments only (toxicity baseline);
stage2 = Civil Comments -> EXIST fine-tuned. Threshold for the *_pred columns is 0.5."""
import sys, os, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
inp = sys.argv[1] if len(sys.argv) > 1 else 'eval_set_raw.csv'
out = sys.argv[2] if len(sys.argv) > 2 else 'eval_set_scored.csv'
CKPTS = {'distilbert_stage1': 'checkpoints/stage1', 'distilbert_stage2': 'checkpoints/stage2',
         'bert_stage1': 'checkpoints_bert/stage1', 'bert_stage2': 'checkpoints_bert/stage2',
         'roberta_stage1': 'checkpoints_roberta/stage1', 'roberta_stage2': 'checkpoints_roberta/stage2'}
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
df = pd.read_csv(inp); texts = df.text.astype(str).tolist()
for name, path in CKPTS.items():
    if not os.path.exists(path + '/config.json'):
        print('skip (missing)', name); continue
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    if m.config.num_labels != 2:
        print('skip (not binary)', name); continue
    P = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i+32], truncation=True, max_length=128, padding=True, return_tensors='pt').to(dev)
            P.append(torch.softmax(m(**b).logits, -1)[:, 1].cpu().numpy())
    df['p_' + name] = np.concatenate(P).round(4); df[name + '_pred'] = (df['p_' + name] >= .5).astype(int)
    print(f'{name}: flagged sexist {df[name + "_pred"].mean():.1%}')
df.to_csv(out, index=False); print('saved', out, df.shape)
