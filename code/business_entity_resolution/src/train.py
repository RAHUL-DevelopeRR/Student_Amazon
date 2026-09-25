"""Entity-disjoint learned matcher pilot; no test predictions or external data."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from config import ROOT,connect,save_json,log_experiment
from metric import read_mapping,entity_f05
from features import pair_features


def fold(ident):
    bucket=int(hashlib.sha256(('42:'+ident).encode()).hexdigest()[:8],16)%10
    return 'train' if bucket<6 else 'calibration' if bucket<8 else 'evaluation'


def score(ids, truth, candidates, scores, threshold):
    vals=[]; singleton=[]; non=[]; tp=fp=fn=0
    for ident in ids:
        pred={t for t,p in zip(candidates[ident],scores[ident]) if p>=threshold}
        t=truth[ident]; value=entity_f05(t,pred)
        vals.append(value); (non if t else singleton).append(value)
        tp+=len(t & pred);fp+=len(pred-t);fn+=len(t-pred)
    return {'queries':len(ids),'macro_f05':float(np.mean(vals)),
            'singleton_accuracy':float(np.mean(singleton)) if singleton else None,
            'non_singleton_f05':float(np.mean(non)) if non else None,
            'pair_precision':tp/(tp+fp) if tp+fp else 0,
            'pair_recall':tp/(tp+fn) if tp+fn else 0,
            'tp':tp,'fp':fp,'fn':fn}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-dir',type=Path,default=ROOT/'artifacts/blocking_pilot')
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/matcher_pilot')
    a=p.parse_args();start=time.monotonic()
    meta=json.loads((a.candidate_dir/'metrics.json').read_text())
    candidates={k:sorted(v) for k,v in read_mapping(a.candidate_dir/'candidate_pairs.tsv','candidate_entity_ids').items()}
    print(f'Loading source records for {len(candidates):,} query IDs',flush=True)
    con=connect()
    con.execute('CREATE TEMP TABLE selected_ids AS SELECT unnest(?) entity_id',[list(candidates)])
    queries={r[0]:r for r in con.execute("SELECT s.entity_id,s.name_norm,s.address_norm,coalesce(s.country,'') FROM train_source1 s JOIN selected_ids q USING(entity_id)").fetchall()}
    target_ids=sorted({t for c in candidates.values() for t in c})
    print(f'Loading {len(target_ids):,} distinct candidate target records',flush=True)
    con.execute('CREATE TEMP TABLE selected_targets AS SELECT unnest(?) entity_id',[target_ids])
    targets={r[0]:r for r in con.execute("SELECT t.entity_id,t.name_norm,t.address_norm,coalesce(t.country,'') FROM targets t JOIN selected_targets q USING(entity_id)").fetchall()}
    print('Loading training labels for evaluation and fitting',flush=True)
    truth={i:set(v.split(',')) if v else set() for i,v in con.execute('SELECT source1_entity_id,matched_entity_ids FROM truth JOIN selected_ids ON source1_entity_id=entity_id').fetchall()}
    con.close()
    if len(queries)!=len(candidates) or len(truth)!=len(candidates) or len(targets)!=len(target_ids):
        raise ValueError('Missing IDs in input corpus')
    # Group duplicate normalized name+address identities together, beyond ID split.
    groups={i:fold(queries[i][1]+'|'+queries[i][2]+'|'+queries[i][3]) for i in candidates}
    matrices={};labels={}
    print(f'Extracting features for {len(candidates):,} queries / {sum(map(len,candidates.values())):,} pairs',flush=True)
    for i,c in candidates.items():
        matrices[i]=np.asarray([pair_features(queries[i],targets[t]) for t in c],dtype=np.float32).reshape(-1,27)
        labels[i]=np.asarray([t in truth[i] for t in c],dtype=np.int8)
    train_ids=[i for i in candidates if groups[i]=='train']
    cal_ids=[i for i in candidates if groups[i]=='calibration']
    eval_ids=[i for i in candidates if groups[i]=='evaluation']
    if not train_ids or not cal_ids or not eval_ids:
        raise ValueError('Need nonempty train/calibration/evaluation groups')
    x=np.concatenate([matrices[i] for i in train_ids]);y=np.concatenate([labels[i] for i in train_ids])
    print(f'Training on {len(train_ids):,} identities / {len(y):,} pairs; calibration={len(cal_ids)}, evaluation={len(eval_ids)}',flush=True)
    model=lgb.LGBMClassifier(n_estimators=180,num_leaves=15,max_depth=6,
                             min_child_samples=40,learning_rate=.05,verbosity=-1,
                             random_state=42,n_jobs=2,deterministic=True,force_col_wise=True)
    model.fit(x,y)
    ordered=list(matrices)
    all_x=np.concatenate([matrices[i] for i in ordered])
    all_probabilities=model.predict_proba(all_x)[:,1]
    offsets=np.cumsum([0]+[len(matrices[i]) for i in ordered])
    probabilities={i:all_probabilities[offsets[j]:offsets[j+1]] for j,i in enumerate(ordered)}
    thresholds=np.linspace(.05,.99,95)
    threshold=max(thresholds,key=lambda t:(score(cal_ids,truth,candidates,probabilities,t)['macro_f05'],t))
    result={'scope':meta['scope'],'warning':'Reduced-corpus pilot is optimistic; not a leaderboard estimate' if 'pilot' in meta['scope'] else 'Sampled-query assessment; retrieval policy still requires production validation',
            'candidate_metrics':meta,'threshold':float(threshold),
            'split':'SHA256(seed + normalized name/address/country identity), 60/20/20; evaluation not used in fitting or tuning',
            'train_queries':len(train_ids),'train_pairs':len(y),'train_positive_pairs':int(y.sum()),
            'calibration':score(cal_ids,truth,candidates,probabilities,threshold),
            'evaluation':score(eval_ids,truth,candidates,probabilities,threshold),
            'evaluation_retrieval_missed_links':sum(len(truth[i]-set(candidates[i])) for i in eval_ids),
            'evaluation_oracle_macro_f05_ceiling':float(np.mean([entity_f05(truth[i],truth[i]&set(candidates[i])) for i in eval_ids])),
            'all_empty_evaluation_macro_f05':sum(not truth[i] for i in eval_ids)/len(eval_ids),
            'evaluation_by_country':{},'seconds':time.monotonic()-start}
    for country in sorted({queries[i][3] for i in eval_ids}):
        ids=[i for i in eval_ids if queries[i][3]==country]
        result['evaluation_by_country'][country]=score(ids,truth,candidates,probabilities,threshold)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    model.booster_.save_model(str(a.output_dir/'model.txt'))
    restored=lgb.Booster(model_file=str(a.output_dir/'model.txt'))
    if not np.allclose(restored.predict(all_x[:1000],num_threads=2),all_probabilities[:1000],atol=1e-12):
        raise RuntimeError('Saved model predictions differ after reload')
    save_json(a.output_dir/'model_config.json',{'threshold':float(threshold),'feature_count':27,
        'feature_function':'features.pair_features','model_type':'LightGBM binary classifier',
        'license':'MIT (LightGBM)','scope':meta['scope'],'seed':42,
        'parameters':model.get_params(),'candidate_file_sha256':hashlib.sha256((a.candidate_dir/'candidate_pairs.tsv').read_bytes()).hexdigest()})
    save_json(a.output_dir/'metrics.json',result)
    save_json(a.output_dir/'splits.json',groups)
    errors=[]
    for i in eval_ids:
        pred={t for t,s in zip(candidates[i],probabilities[i]) if s>=threshold}
        if pred!=truth[i]:
            errors.append({'source1':i,'country':queries[i][3],'false_positive':sorted(pred-truth[i]),'false_negative':sorted(truth[i]-pred)})
    save_json(a.output_dir/'evaluation_errors.json',errors)
    log_experiment('matcher',result['evaluation'],{'scope':meta['scope'],'threshold':float(threshold)})
    print(json.dumps({k:v for k,v in result.items() if k!='candidate_metrics'},indent=2))


if __name__=='__main__':
    main()
