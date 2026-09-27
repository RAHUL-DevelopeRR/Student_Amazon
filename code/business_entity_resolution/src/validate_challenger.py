"""One-shot new holdout, selected solely on development calibration."""
import csv,json,time
import numpy as np
import lightgbm as lgb
from config import ROOT,save_json
from build_test_candidates import Retriever
from challenger_features import enhanced
from features import pair_features
from probe_joint_retrieval import joint
from metric import entity_f05
from inference_common import sha256


def main():
    out=ROOT/'artifacts/challenger';dest=out/'fresh_validation.json'
    if dest.exists():raise ValueError('Do not tune or overwrite the fresh holdout')
    reports=json.loads((out/'results.json').read_text());chosen=max(reports,key=lambda r:r['calibration']['macro_f05'])
    save_json(out/'chosen.json',chosen)
    modelpath=out/(chosen['name']+'.txt');model=lgb.Booster(model_file=str(modelpath))
    baseline=lgb.Booster(model_file=str(ROOT/'artifacts/baseline-v1/model.txt'))
    r=Retriever(ROOT/'artifacts/train_inference_index');old=r.query_rows(0,4000);qs=r.query_rows(4000,1000)
    oldgroups={q[1:] for q in old};assert not oldgroups&{q[1:] for q in qs}
    selected={q[0] for q in qs};truth={}
    with (ROOT/'dataset/train/train_ground_truth.tsv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['source1_entity_id'] in selected:truth[row['source1_entity_id']]=set(filter(None,row['matched_entity_ids'].split(',')))
    stats={name:{'scores':[],'tp':0,'fp':0,'fn':0,'found':0,'pairs':0} for name in ['baseline','challenger']};start=time.monotonic()
    for j,q in enumerate(qs):
        baseids=r.retrieve(q);allids=sorted(set(baseids)|set(joint(r,q)));targets=r.target_rows(allids);tr=truth[q[0]]
        for name,ids,m,fn,threshold in [('baseline',baseids,baseline,pair_features,.48),('challenger',allids,model,enhanced,chosen['threshold'])]:
            x=np.asarray([fn(q,targets[i]) for i in ids],dtype='float32').reshape(-1,m.num_feature());p=m.predict(x,num_threads=2) if len(x) else []
            cand={targets[i][0] for i in ids};pred={targets[i][0] for i,prob in zip(ids,p) if prob>=threshold};s=stats[name]
            s['scores'].append(entity_f05(tr,pred));s['tp']+=len(pred&tr);s['fp']+=len(pred-tr);s['fn']+=len(tr-pred);s['found']+=len(cand&tr);s['pairs']+=len(ids)
        if j%100==0:print('fresh',j,flush=True)
    total=sum(map(len,truth.values()));result={}
    for name,s in stats.items():
        result[name]={'macro_f05':float(np.mean(s['scores'])),'pair_precision':s['tp']/max(s['tp']+s['fp'],1),'pair_recall':s['tp']/total,'candidate_recall':s['found']/total,'pairs':s['pairs']}
    result.update(scope='untouched_1000_queries_4001_to_5000_full_target_corpus',targets=r.meta['targets'],chosen=chosen['name'],threshold=chosen['threshold'],model_sha256=sha256(modelpath),seconds=time.monotonic()-start,seed=42,tuning_after_result=False)
    save_json(dest,result);print(json.dumps(result),flush=True);r.close()


if __name__=='__main__':main()
