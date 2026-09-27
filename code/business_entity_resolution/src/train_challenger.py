"""Time-bounded challenger on existing development queries, never the fresh holdout."""
import json,time,csv
from pathlib import Path
import numpy as np
import lightgbm as lgb
from config import ROOT,save_json
from inference_common import sqlite
from metric import read_mapping
from train import fold,score
from challenger_features import enhanced


def main():
    start=time.monotonic();out=ROOT/'artifacts/challenger';out.mkdir(exist_ok=True)
    cand={q:sorted(v) for q,v in read_mapping(ROOT/'artifacts/blocking_token_wide/candidate_pairs.tsv','candidate_entity_ids').items()}
    db=sqlite(ROOT/'artifacts/train_inference_index/records.sqlite',True)
    queries={q:db.execute('SELECT entity_id,name,address,country FROM queries WHERE entity_id=?',(q,)).fetchone() for q in cand}
    truth={}
    with (ROOT/'dataset/train/train_ground_truth.tsv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['source1_entity_id'] in cand:truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
    ids=list(cand);offset=np.cumsum([0]+[len(cand[q]) for q in ids]);cache=out/'features.npy'
    if not cache.exists():
        tmp=out/'features.tmp.npy';x=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(int(offset[-1]),51))
        for j,q in enumerate(ids):
            targets={}
            for k in range(0,len(cand[q]),500):
                batch=cand[q][k:k+500]
                for r in db.execute('SELECT entity_id,name,address,country FROM targets WHERE entity_id IN ('+','.join('?' for _ in batch)+')',batch):targets[r[0]]=r
            x[offset[j]:offset[j+1]]=np.asarray([enhanced(queries[q],targets[t]) for t in cand[q]],dtype='float32').reshape(-1,51)
            if j%250==0:print('features',j,flush=True)
        x.flush();del x;tmp.replace(cache)
    db.close();x=np.load(cache,mmap_mode='r')
    groups={q:fold('|'.join(queries[q][1:])) for q in ids}
    trainmask=np.repeat([groups[q]=='train' for q in ids],np.diff(offset))
    y=np.asarray([t in truth[q] for q in ids for t in cand[q]],dtype='int8')
    cal=[q for q in ids if groups[q]=='calibration'];ev=[q for q in ids if groups[q]=='evaluation']
    results=[]
    for name,n,leaves in [('numeric',300,31),('numeric_capacity',500,63)]:
        m=lgb.LGBMClassifier(n_estimators=n,num_leaves=leaves,max_depth=-1,min_child_samples=30,learning_rate=.05,verbosity=-1,random_state=42,n_jobs=2,deterministic=True,force_col_wise=True)
        m.fit(x[trainmask],y[trainmask]);p=m.predict_proba(x)[:,1]
        probs={q:p[offset[j]:offset[j+1]] for j,q in enumerate(ids)}
        threshold=max(np.arange(.15,.86,.025),key=lambda t:(score(cal,truth,cand,probs,t)['macro_f05'],t))
        report={'name':name,'threshold':float(threshold),'calibration':score(cal,truth,cand,probs,threshold),'development_evaluation':score(ev,truth,cand,probs,threshold),'seed':42,'pairs':int(offset[-1]),'target_population':10320219,'seconds':time.monotonic()-start,'parameters':m.get_params(),'feature_count':51,'note':'Existing development evaluation; not untouched and not a leaderboard estimate.'}
        m.booster_.save_model(str(out/(name+'.txt')));save_json(out/(name+'.json'),report);results.append(report)
        print(json.dumps(report),flush=True)
    save_json(out/'results.json',results)


if __name__=='__main__':main()
