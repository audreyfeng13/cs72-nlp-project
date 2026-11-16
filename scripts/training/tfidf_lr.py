import pandas as pd, numpy as np, time, os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

J='jigsaw-unintended-bias-in-toxicity-classification/train.csv'
E='EXIST2021-2025_datasets/2021 EXIST/'
j=pd.read_csv(J,usecols=['comment_text','target']).dropna()
j['y']=(j.target>=0.5).astype(int)
def load(p):
    d=pd.read_csv(E+p,sep='\t'); d=d[(d.language=='en')&(d.source=='twitter')].copy()
    d['y']=(d.task1=='sexist').astype(int); return d
et,ee=load('training/EXIST2021_training.tsv'),load('test/EXIST2021_test_labeled.tsv')
print("EXIST en+twitter: train",et.shape,"test",ee.shape,"| sexist rate train %.3f test %.3f"%(et.y.mean(),ee.y.mean()))
print("Jigsaw",j.shape,"toxic rate %.3f"%j.y.mean())

Xtr,Xva,ytr,yva=train_test_split(j.comment_text,j.y,test_size=0.1,random_state=0,stratify=j.y)
t=time.time()
vec=TfidfVectorizer(max_features=200000,ngram_range=(1,2),min_df=5,sublinear_tf=True,lowercase=True)
A=vec.fit_transform(Xtr); print("vectorized",A.shape,"%.0fs"%(time.time()-t))
clf=LogisticRegression(C=4,solver='liblinear',class_weight='balanced',max_iter=200).fit(A,ytr)
print("trained %.0fs"%(time.time()-t))
os.makedirs("models",exist_ok=True); joblib.dump({"vec":vec,"clf":clf},"models/tfidf_lr.joblib"); print("saved models/tfidf_lr.joblib")

def ev(name,X,y):
    p=clf.predict_proba(vec.transform(X))[:,1]; pr=(p>=.5).astype(int)
    print(f"\n=== {name} (n={len(y)}) AUC={roc_auc_score(y,p):.4f} macroF1={f1_score(y,pr,average='macro'):.4f}")
    print(confusion_matrix(y,pr)); print(classification_report(y,pr,digits=3))
ev("Jigsaw held-out val (toxic)",Xva,yva)
ev("EXIST en/twitter TRAIN (sexist)",et.text,et.y)
ev("EXIST en/twitter TEST (sexist)",ee.text,ee.y)

fn=np.array(vec.get_feature_names_out()); c=clf.coef_[0]
print("\nTop toxic features:",list(fn[np.argsort(-c)[:20]]))
print("Top non-toxic features:",list(fn[np.argsort(c)[:20]]))
