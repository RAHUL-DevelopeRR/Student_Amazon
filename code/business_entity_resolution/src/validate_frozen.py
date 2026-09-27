"""Explicit TRAIN-only frozen-policy evaluation. Never imported by test inference."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from config import ROOT,save_json
from inference_common import read_json,sha256,sqlite
from metric import entity_f05,read_mapping
from predict import run
from build_test_candidates import Retriever


def evaluate(index,baseline,output,truth_path,limit=1000):
    index,baseline,output=map(Path,[index,baseline,output]);output.mkdir(parents=True,exist_ok=True)
    meta=read_json(index/'manifest.json')
    if meta['signature']['split']!='train': raise ValueError('Validation requires a separate training index')
    audit=read_json(ROOT/'artifacts/data_profile.json')
    for s in [1,2,3]:
        if meta['signature']['sources'][f'train_source{s}.tsv']!=audit['files'][f'train_source{s}']['sha256']:
            raise ValueError('Training sources differ from the audited identity graph')
    if sha256(truth_path)!=audit['ground_truth']['sha256'] or audit['ground_truth']['targets_with_multiple_owners']:
        raise ValueError('Identity ownership is not verified; connected-group splitting required')
    report=output/'validation.json'
    if report.exists():
        raise ValueError('Untouched result already recorded; do not tune or overwrite this evaluation')
    records=sqlite(index/'records.sqlite',True)
    dev=records.execute('SELECT entity_id,name,address,country FROM queries WHERE ordinal<=3000').fetchall()
    expected=set(read_mapping(ROOT/'artifacts/blocking_token_wide/candidate_pairs.tsv','candidate_entity_ids'))
    if {r[0] for r in dev}!=expected: raise ValueError('Development query selection differs')
    fresh=records.execute('SELECT entity_id,name,address,country FROM queries WHERE ordinal>3000 AND ordinal<=? ORDER BY ordinal',(3000+limit,)).fetchall()
    if {r[1:] for r in dev}&{r[1:] for r in fresh}: raise ValueError('Normalized identity overlaps development group')
    records.close()
    selected={r[0] for r in fresh};truth={}
    with Path(truth_path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for r in reader:
            if r['source1_entity_id'] in selected:
                truth[r['source1_entity_id']]=set(r['matched_entity_ids'].split(',')) if r['matched_entity_ids'] else set()
    if set(truth)!=selected: raise ValueError('Missing training labels')
    result=run(index,baseline,output/'inference',start=3000,limit=limit)
    by_country={r[0]:r[3] for r in fresh};scores=[];single=[];tp=fp=fn=found=positives=0;country=defaultdict(list)
    for shard in result['shards']:
        candidates=read_mapping(shard['scored'],'candidate_entity_ids');pred=read_mapping(shard['matching'])
        for q,p in pred.items():
            t=truth[q];v=entity_f05(t,p);scores.append(v);country[by_country[q]].append(v)
            if not t:single.append(v)
            tp+=len(t&p);fp+=len(p-t);fn+=len(t-p);found+=len(t&candidates[q]);positives+=len(t)
    metrics={'scope':'untouched_training_queries_full_target_corpus','queries':len(scores),
             'macro_f05':sum(scores)/len(scores),'candidate_recall':found/positives,
             'pair_precision':tp/(tp+fp) if tp+fp else 0,'pair_recall':tp/(tp+fn),
             'singleton_accuracy':sum(single)/len(single) if single else None,'singleton_queries':len(single),
             'by_country':{c:sum(v)/len(v) for c,v in country.items()},'tp':tp,'fp':fp,'fn':fn,
             'candidates_per_query':result['average_candidates'],'policy_sha256':sha256(output/'inference/run.json'),
             'baseline_sha256':sha256(baseline/'manifest.json'),'truth_sha256':sha256(truth_path),
             'inference_code':result['identity']['code'],
             'start_ordinal':3001,'last_ordinal':3000+limit,'tuning_after_result':False}
    save_json(report,metrics);return metrics


def cascade(index,baseline,output,truth_path):
    """Measure exact-name+address on development queries without enabling bypass."""
    index,baseline,output=map(Path,[index,baseline,output]);r=Retriever(index)
    if r.meta['signature']['split']!='train': raise ValueError('TRAIN only')
    queries=r.query_rows(0,3000);wanted={q[0] for q in queries};truth={}
    with Path(truth_path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['source1_entity_id'] in wanted: truth[row['source1_entity_id']]=set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
    pred={};candidates={}
    # Use the same frozen pipeline on development data, never on the fresh holdout.
    runinfo=run(index,baseline,output/'dev_inference',limit=3000)
    for shard in runinfo['shards']:
        pred.update(read_mapping(shard['matching']));candidates.update(read_mapping(shard['scored'],'candidate_entity_ids'))
    previous=read_mapping(ROOT/'artifacts/blocking_token_wide/candidate_pairs.tsv','candidate_entity_ids')
    parity={'queries':len(candidates),'identical_candidate_sets':sum(candidates[q]==previous[q] for q in candidates),
            'added_pairs':sum(len(candidates[q]-previous[q]) for q in candidates),
            'removed_pairs':sum(len(previous[q]-candidates[q]) for q in candidates)}
    tp=fp=eligible=0;before=after=0
    for q in queries:
        a=r.cached(0,1,q[1]) if q[1] else None;b=r.cached(1,1,q[2]) if q[2] else None
        ids=sorted(set(map(int,a[1]))&set(map(int,b[1]))) if a and b else []
        targets=r.target_rows(ids)
        rule={v[0] for v in targets.values() if (not q[3] or not v[3] or q[3].lower()==v[3].lower()) and v[0] in candidates[q[0]]}
        tp+=len(rule&truth[q[0]]);fp+=len(rule-truth[q[0]]);eligible+=len(rule)
        before+=entity_f05(truth[q[0]],pred[q[0]]);after+=entity_f05(truth[q[0]],pred[q[0]]|rule)
    r.close()
    report={'scope':'development_3000_full_target_corpus','rule':'nonempty exact normalized name AND address, compatible country, within scored candidates',
            'eligible_pairs':eligible,'tp':tp,'fp':fp,'precision':tp/eligible if eligible else None,
            'baseline_macro_f05':before/3000,'cascade_macro_f05':after/3000,'macro_f05_delta':(after-before)/3000,
            'enabled':False,'retrieval_parity_with_previous_SQL_run':parity,
            'decision':'Measure only; retain frozen model scoring for the first submission.'}
    save_json(output/'cascade.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,default=ROOT/'artifacts/train_inference_index')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/baseline-v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/frozen_validation')
    p.add_argument('--truth',type=Path,default=ROOT/'dataset/train/train_ground_truth.tsv')
    p.add_argument('--cascade',action='store_true');p.add_argument('--limit',type=int,default=1000);a=p.parse_args()
    result=cascade(a.index,a.baseline,a.output_dir,a.truth) if a.cascade else evaluate(a.index,a.baseline,a.output_dir,a.truth,a.limit)
    print(json.dumps(result,indent=2))
