"""Resumable bounded-batch inference from a frozen model; never reads labels."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from config import ROOT,save_json
from features import pair_features
from postprocess import select_matches
from build_test_candidates import Retriever,candidate_shard
from inference_common import sha256,code_hash,read_json,seal,valid,freeze,sampled_rss


def score_shard(candidate,directory,records,model,threshold,model_hash):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    stem=Path(candidate).name.split('.')[0]
    matching=directory/(stem+'.matching.tsv');scored=directory/(stem+'.scored.tsv')
    identity={'candidate_sha256':sha256(candidate),'model':model_hash,'threshold':threshold,
              'code':code_hash(['features.py','normalize.py','postprocess.py','predict.py'])}
    if valid(matching,identity) and valid(scored,identity): return matching,scored
    stats={k:0.0 for k in ['candidate_loading_seconds','feature_seconds','scoring_seconds','output_seconds']}
    total=time.monotonic();pairs=rows=0
    mt=matching.with_suffix('.tmp');ct=scored.with_suffix('.tmp')
    with Path(candidate).open(encoding='utf-8') as inp,mt.open('w',encoding='utf-8',newline='') as mf,ct.open('w',encoding='utf-8',newline='') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n');cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
        while True:
            start=time.monotonic();batch=[]
            # Bound candidate strings and feature matrices to 32 queries.
            for _ in range(32):
                line=inp.readline()
                if not line: break
                batch.append(json.loads(line))
            if not batch: break
            ids=sorted({i for _,ids in batch for i in ids});targets=records.target_rows(ids)
            if len(targets)!=len(ids): raise ValueError('Candidate target absent from index')
            stats['candidate_loading_seconds']+=time.monotonic()-start
            start=time.monotonic()
            x=np.asarray([pair_features(q,targets[i]) for q,ids in batch for i in ids],dtype=np.float32).reshape(-1,27)
            stats['feature_seconds']+=time.monotonic()-start
            start=time.monotonic();prob=model.predict(x,num_threads=2) if len(x) else np.empty(0)
            stats['scoring_seconds']+=time.monotonic()-start
            start=time.monotonic();offset=0
            for q,ids in batch:
                entity_ids=[targets[i][0] for i in ids]
                matches=select_matches(entity_ids,prob[offset:offset+len(ids)],threshold)
                if not set(matches)<=set(entity_ids): raise ValueError('Unscored prediction')
                cw.writerow([q[0],','.join(sorted(entity_ids))]);mw.writerow([q[0],','.join(sorted(matches))])
                rows+=1;pairs+=len(ids);offset+=len(ids)
            stats['output_seconds']+=time.monotonic()-start
    mt.replace(matching);ct.replace(scored)
    metadata=dict(identity,**stats,seconds=time.monotonic()-total,rows=rows,pairs=pairs,sampled_rss=sampled_rss())
    seal(matching,metadata);seal(scored,metadata)
    return matching,scored


def run(index,baseline,output,start=0,limit=10000,shard_size=250):
    index,baseline,output=map(Path,[index,baseline,output]);output.mkdir(parents=True,exist_ok=True)
    config=read_json(baseline/'model_config.json');frozen=read_json(baseline/'manifest.json')
    for name,h in frozen['identity']['files'].items():
        if sha256(baseline/name)!=h: raise ValueError('Frozen baseline corrupted')
    if frozen['identity']['code']!=code_hash(['features.py','normalize.py']): raise ValueError('Frozen feature code changed')
    model=lgb.Booster(model_file=str(baseline/'model.txt'))
    if model.num_feature()!=27 or config['feature_count']!=27: raise ValueError('Wrong model schema')
    retriever=Retriever(index)
    if limit<=0 or shard_size<=0 or start<0 or start+limit>retriever.meta['queries']: raise ValueError('Invalid inference range')
    identity={'index_sha256':sha256(index/'manifest.json'),'baseline_sha256':sha256(baseline/'manifest.json'),
              'start':start,'limit':limit,'shard_size':shard_size,
              'code':code_hash(['build_test_candidates.py','predict.py','postprocess.py','features.py','normalize.py'])}
    runpath=output/'run.json'
    if runpath.exists() and read_json(runpath)['identity']!=identity: raise ValueError('Run configuration changed; choose a new output directory')
    save_json(runpath,{'identity':identity,'status':'running'})
    counts=[];times={k:0.0 for k in ['candidate_generation_seconds','candidate_retrieval_seconds','candidate_output_seconds','candidate_loading_seconds','feature_seconds','scoring_seconds','output_seconds','inference_seconds']};memory=[];shards=[]
    begin=time.monotonic()
    try:
        for offset in range(start,start+limit,shard_size):
            n=min(shard_size,start+limit-offset)
            print(f'SHARD {offset:,}..{offset+n:,}',flush=True)
            candidate=candidate_shard(retriever,output/'candidates',offset,n)
            matching,scored=score_shard(candidate,output/'scored',retriever,model,config['threshold'],sha256(baseline/'model.txt'))
            cm=read_json(candidate.with_suffix(candidate.suffix+'.json'));sm=read_json(matching.with_suffix(matching.suffix+'.json'))
            counts.extend(cm['counts']);times['candidate_generation_seconds']+=cm['seconds'];times['inference_seconds']+=sm['seconds']
            times['candidate_retrieval_seconds']+=cm['retrieval_seconds'];times['candidate_output_seconds']+=cm['output_seconds']
            for k in ['candidate_loading_seconds','feature_seconds','scoring_seconds','output_seconds']:times[k]+=sm[k]
            memory.extend([cm.get('sampled_rss'),sm.get('sampled_rss')])
            shards.append({'start':offset,'count':n,'candidate':str(candidate.resolve()),'matching':str(matching.resolve()),'scored':str(scored.resolve())})
        total=times['candidate_generation_seconds']+times['inference_seconds']
        result={'identity':identity,'status':'complete','shards':shards,'queries':limit,'pairs':sum(counts),
                'average_candidates':sum(counts)/limit,'candidate_quantiles_p50_p90_p99':np.quantile(counts,[.5,.9,.99]).tolist(),
                **times,'total_seconds':total,'this_invocation_seconds':time.monotonic()-begin,
                'sampled_memory_max_bytes':max([m for m in memory if m is not None],default=None),
                'output_bytes':sum(Path(s[k]).stat().st_size for s in shards for k in ['matching','scored']),
                'projected_full_seconds':total*retriever.meta['queries']/limit,
                'projection_note':'Linear estimate excludes one-time index build, assembly and final validator; allow substantial headroom.'}
        save_json(runpath,result);return result
    finally: retriever.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,default=ROOT/'artifacts/test_index')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/baseline-v1')
    p.add_argument('--freeze-from',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/test_pilot_10k')
    p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=10000);p.add_argument('--shard-size',type=int,default=250)
    a=p.parse_args()
    if a.freeze_from: freeze(a.freeze_from,a.baseline)
    if a.limit>10000 and read_json(a.index/'manifest.json')['signature']['split']=='test':
        from submission_gate import assess
        gate=assess(ROOT/'artifacts/test_pilot_10k/run.json',ROOT/'artifacts/frozen_validation/validation.json',a.index)
        save_json(ROOT/'artifacts/submission_gate.json',gate)
        if not gate['full_run_allowed']:raise ValueError('Full-run feasibility gate failed; inspect artifacts/submission_gate.json')
    print(json.dumps(run(a.index,a.baseline,a.output_dir,a.start,a.limit,a.shard_size),indent=2))


if __name__=='__main__':main()
