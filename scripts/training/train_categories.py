"""Multi-label category head (EXIST task2, English) on top of checkpoints/stage2.
Five independent sigmoid outputs; non-sexist rows are all-zero. Threshold 0.5."""
import time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, f1_score
torch.manual_seed(0); np.random.seed(0)
CATS=["ideological-inequality","stereotyping-dominance","objectification","sexual-violence","misogyny-non-sexual-violence"]
SRC,OUT='checkpoints/stage2','checkpoints/stage2_categories'
EPOCHS,LR,BS,MAXLEN=3,2e-5,32,128
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); print('device',dev)
E='EXIST2021-2025_datasets/2021 EXIST/'
def load(p):
    d=pd.read_csv(E+p,sep='\t'); d=d[d.language=='en'].reset_index(drop=True)
    Y=np.stack([(d.task2==c).astype(np.float32).values for c in CATS],1); return d,Y
dtr,Ytr=load('training/EXIST2021_training.tsv'); dte,Yte=load('test/EXIST2021_test_labeled.tsv')
print('train',dtr.shape,'test',dte.shape,'| label cardinality (train) %.3f'%Ytr.sum(1).mean(),'| rows with >1 label:',int((Ytr.sum(1)>1).sum()))
tok=AutoTokenizer.from_pretrained(SRC)
model=AutoModelForSequenceClassification.from_pretrained(SRC,num_labels=5,problem_type='multi_label_classification',
    ignore_mismatched_sizes=True,id2label=dict(enumerate(CATS)),label2id={c:i for i,c in enumerate(CATS)}).to(dev)
def loader(texts,Y,shuffle):
    enc=tok(list(texts),truncation=True,max_length=MAXLEN)
    items=[{'input_ids':enc['input_ids'][i],'attention_mask':enc['attention_mask'][i],'labels':Y[i].tolist()} for i in range(len(Y))]
    def coll(b):
        x=tok.pad([{k:v for k,v in i.items() if k!='labels'} for i in b],return_tensors='pt'); x['labels']=torch.tensor([i['labels'] for i in b],dtype=torch.float32); return x
    return DataLoader(items,batch_size=BS,shuffle=shuffle,collate_fn=coll)
dl=loader(dtr.text,Ytr,True)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01); steps=EPOCHS*len(dl)
sch=get_linear_schedule_with_warmup(opt,int(.06*steps),steps)
for ep in range(EPOCHS):
    model.train(); t=time.time(); tot=0
    for b in dl:
        b={k:v.to(dev) for k,v in b.items()}; loss=model(**b).loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad(); tot+=loss.item()
    print(f'epoch {ep+1} mean BCE loss {tot/len(dl):.4f} ({time.time()-t:.0f}s)',flush=True)
model.eval(); P=[]
with torch.no_grad():
    for b in loader(dte.text,Yte,False):
        b={k:v.to(dev) for k,v in b.items() if k!='labels'}; P.append(torch.sigmoid(model(**b).logits).cpu().numpy())
P=np.concatenate(P); Pr=(P>=.5).astype(int)
def report(name,m):
    p,r,f,s=precision_recall_fscore_support(Yte[m],Pr[m],zero_division=0)
    print(f'\n=== {name} (n={m.sum()}) | macro-F1 over 5 categories = {f.mean():.4f} | macro-P {p.mean():.3f} macro-R {r.mean():.3f}')
    print(f'{"category":<32}{"precision":>10}{"recall":>8}{"F1":>8}{"support":>9}')
    for i,c in enumerate(CATS): print(f'{c:<32}{p[i]:>10.3f}{r[i]:>8.3f}{f[i]:>8.3f}{int(s[i]):>9}')
report('EXIST en TEST, all rows',np.ones(len(dte),bool))
report('EXIST en TEST, sexist rows only',(Yte.sum(1)>0))
print('\npredicted-label cardinality on test: %.3f | rows with >1 predicted label: %d | rows with none: %d'%(Pr.sum(1).mean(),(Pr.sum(1)>1).sum(),(Pr.sum(1)==0).sum()))
model.save_pretrained(OUT); tok.save_pretrained(OUT); print('saved',OUT)
