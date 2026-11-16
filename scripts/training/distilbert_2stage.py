"""Two-stage DistilBERT fine-tuning: Civil Comments (toxicity) -> EXIST 2021 (sexism, English).
Usage: python scripts/training/distilbert_2stage.py [--model distilbert-base-uncased] [--n-jigsaw 50000] [--smoke] [--twitter-only]"""
import argparse, time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

ap=argparse.ArgumentParser()
ap.add_argument('--n-jigsaw',type=int,default=50000)
ap.add_argument('--epochs1',type=int,default=1); ap.add_argument('--epochs2',type=int,default=3)
ap.add_argument('--lr1',type=float,default=2e-5); ap.add_argument('--lr2',type=float,default=1e-5)
ap.add_argument('--bs',type=int,default=32); ap.add_argument('--max-len',type=int,default=128)
ap.add_argument('--skip-stage1',action='store_true',help='baseline: EXIST-only fine-tuning')
ap.add_argument('--init-from',default=None,help='checkpoint dir to start from (use with --skip-stage1)')
ap.add_argument('--train-file',default=None,help='alternate EXIST-format TSV for stage 2 training')
ap.add_argument('--twitter-only',action='store_true'); ap.add_argument('--smoke',action='store_true')
ap.add_argument('--model',default='distilbert-base-uncased'); ap.add_argument('--out',default='checkpoints'); ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); print('device',dev)

J='jigsaw-unintended-bias-in-toxicity-classification/train.csv'
E='EXIST2021-2025_datasets/2021 EXIST/'
n=200 if a.smoke else a.n_jigsaw
j=pd.read_csv(J,usecols=['comment_text','target']).dropna().sample(n,random_state=a.seed)
j['y']=(j.target>=0.5).astype(int); print('Jigsaw sample',j.shape,'toxic rate %.3f'%j.y.mean())
def load(p):
    d=pd.read_csv(E+p,sep='\t'); d=d[d.language=='en']
    if a.twitter_only: d=d[d.source=='twitter']
    d=d.copy(); d['y']=(d.task1=='sexist').astype(int); return d
et,ee=load('training/EXIST2021_training.tsv'),load('test/EXIST2021_test_labeled.tsv')
if a.train_file:
    et=pd.read_csv(a.train_file,sep='\t'); et=et[et.language=='en'].copy(); et['y']=(et.task1=='sexist').astype(int)
if a.smoke: et,ee=et.head(100),ee.head(100)
print('EXIST en train',et.shape,'test',ee.shape)

tok=AutoTokenizer.from_pretrained(a.init_from or a.model)
model=AutoModelForSequenceClassification.from_pretrained(a.init_from or a.model,num_labels=2).to(dev)

def loader(texts,y,shuffle):
    enc=tok(list(texts),truncation=True,max_length=a.max_len)
    items=[{'input_ids':enc['input_ids'][i],'attention_mask':enc['attention_mask'][i],'labels':int(y[i])} for i in range(len(y))]
    return DataLoader(items,batch_size=a.bs,shuffle=shuffle,collate_fn=lambda b:tok.pad(b,return_tensors='pt'))

@torch.no_grad()
def predict(dl):
    model.eval(); P=[]
    for b in dl:
        b={k:v.to(dev) for k,v in b.items() if k!='labels'}
        P.append(torch.softmax(model(**b).logits,-1)[:,1].cpu().numpy())
    return np.concatenate(P)

def evaluate(name,df):
    dl=loader(df.text if 'text' in df else df.comment_text,df.y.values,False); p=predict(dl); pr=(p>=.5).astype(int)
    print(f'[{name}] n={len(df)} AUC={roc_auc_score(df.y,p):.4f} macroF1={f1_score(df.y,pr,average="macro"):.4f} acc={(pr==df.y.values).mean():.4f}')
    return p

TRAIN_T=0.0
def train(name,df,textcol,epochs,lr):
    global TRAIN_T; t_start=time.time()
    dl=loader(df[textcol],df.y.values,True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    steps=epochs*len(dl); sch=get_linear_schedule_with_warmup(opt,int(.06*steps),steps)
    for ep in range(epochs):
        model.train(); t=time.time(); tot=0
        for i,b in enumerate(dl):
            b={k:v.to(dev) for k,v in b.items()}
            loss=model(**b).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad(); tot+=loss.item()
            if (i+1)%100==0: print(f'  {name} ep{ep+1} step {i+1}/{len(dl)} loss {tot/(i+1):.4f} ({time.time()-t:.0f}s)',flush=True)
        print(f'{name} epoch {ep+1} done, mean loss {tot/len(dl):.4f}, {time.time()-t:.0f}s')
    TRAIN_T+=time.time()-t_start

# Stage 1: Civil Comments
j_tr,j_va=train_test_split(j,test_size=0.1,random_state=a.seed,stratify=j.y)
if not a.skip_stage1:
    train('stage1',j_tr,'comment_text',a.epochs1,a.lr1)
    evaluate('stage1 | Jigsaw val',j_va)
    evaluate('stage1 | EXIST en test (zero-shot)',ee)
    model.save_pretrained(f'{a.out}/stage1'); tok.save_pretrained(f'{a.out}/stage1')

# Stage 2: continue on EXIST (English)
train('stage2',et,'text',a.epochs2,a.lr2)
p=evaluate('stage2 | EXIST en test',ee)
evaluate('stage2 | Jigsaw val (forgetting check)',j_va)
print(classification_report(ee.y,(p>=.5).astype(int),digits=3))
model.save_pretrained(f'{a.out}/stage2'); tok.save_pretrained(f'{a.out}/stage2')
print(f'TOTAL train time (stage1+stage2, excl. eval): {TRAIN_T:.0f}s')
