"""Fixed-step training-speed benchmark (immune to sleep/throttling in full-run wall time).
Usage: python bench_timing.py MODEL [MODEL ...]   (same bs=32, max_len=128 as the real runs)"""
import sys, time, json, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
df = pd.read_csv('jigsaw-unintended-bias-in-toxicity-classification/train.csv', usecols=['comment_text','target']).dropna().sample(32*40, random_state=0)
texts, y = list(df.comment_text), (df.target >= .5).astype(int).tolist()
WARM, STEPS = 5, 30
for name in sys.argv[1:]:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name); m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5); m.train(); times = []
    for i in range(WARM + STEPS):
        s = slice(i*32, i*32+32)
        b = tok(texts[s], truncation=True, max_length=128, padding=True, return_tensors='pt'); b['labels'] = torch.tensor(y[s])
        b = {k: v.to(dev) for k, v in b.items()}
        t = time.time(); loss = m(**b).loss; loss.backward(); opt.step(); opt.zero_grad(); torch.mps.synchronize(); 
        if i >= WARM: times.append(time.time() - t)
    times.sort()
    print(json.dumps({'model': name, 'params_M': round(sum(p.numel() for p in m.parameters())/1e6, 1),
                      'median_s_per_step': round(times[len(times)//2], 3), 'min_s_per_step': round(times[0], 3)}), flush=True)
    del m, opt
