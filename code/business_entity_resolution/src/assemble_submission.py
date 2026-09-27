"""Stream verified scored shards into final TSVs; validate IDs on disk."""
import argparse
import csv
import json
import time
from pathlib import Path
from config import ROOT,save_json
from inference_common import read_json,sha256,valid,sqlite,seal


def assemble(index,run_dir,output,allow_partial=False):
    index,run_dir,output=map(Path,[index,run_dir,output]);output.mkdir(parents=True,exist_ok=True)
    run=read_json(run_dir/'run.json');meta=read_json(index/'manifest.json')
    if run['status']!='complete' or run['identity']['index_sha256']!=sha256(index/'manifest.json'): raise ValueError('Incomplete or mismatched run')
    if not allow_partial and (run['identity']['start']!=0 or run['queries']!=meta['queries']): raise ValueError('Partial run is not a submission')
    if (output/'assembly.json').exists():
        previous=read_json(output/'assembly.json')
        if previous['run_sha256']==sha256(run_dir/'run.json') and previous['partial']==allow_partial:
            if all(sha256(output/name)==v['sha256'] for name,v in previous['files'].items()):return previous
    records=sqlite(index/'records.sqlite',True);started=time.monotonic();count=pairs=matches=empty=0;countries={}
    mpath=output/'matching_results.tsv';cpath=output/'candidate_pairs.tsv'
    mt=mpath.with_suffix('.tmp');ct=cpath.with_suffix('.tmp')
    expected_start=run['identity']['start']
    with mt.open('w',encoding='utf-8',newline='') as mf,ct.open('w',encoding='utf-8',newline='') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n');cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mh=['source1_entity_id','matched_entity_ids'];ch=['source1_entity_id','candidate_entity_ids']
        mw.writerow(mh);cw.writerow(ch)
        for shard in run['shards']:
            if shard['start']!=expected_start: raise ValueError('Missing/overlapping shards')
            for key in ['matching','scored','candidate']:
                if not valid(shard[key],{}): raise ValueError('Corrupt shard: '+key)
            expected=records.execute('SELECT entity_id,country FROM queries WHERE ordinal>? AND ordinal<=? ORDER BY ordinal',(expected_start,expected_start+shard['count'])).fetchall()
            with Path(shard['matching']).open(encoding='utf-8',newline='') as mi,Path(shard['scored']).open(encoding='utf-8',newline='') as ci:
                mr=csv.reader(mi,delimiter='\t');cr=csv.reader(ci,delimiter='\t')
                if next(mr)!=mh or next(cr)!=ch: raise ValueError('Bad output header')
                for q,country in expected:
                    m=next(mr,None);c=next(cr,None)
                    if m is None or c is None or len(m)!=2 or len(c)!=2 or m[0]!=q or c[0]!=q: raise ValueError('S1 coverage/order mismatch')
                    ml=m[1].split(',') if m[1] else [];cl=c[1].split(',') if c[1] else []
                    for ids in [ml,cl]:
                        if len(ids)!=len(set(ids)) or any(not t.startswith(('S2-','S3-')) or 'nan' in t.lower() for t in ids): raise ValueError('Malformed target list')
                    if not set(ml)<=set(cl): raise ValueError('Match outside scored candidates')
                    # Candidate IDs were loaded from the index during scoring. This
                    # fresh lookup independently checks existence, in bounded batches.
                    if cl:
                        found=records.execute('SELECT count(*) FROM targets WHERE entity_id IN ('+','.join('?' for _ in cl)+')',cl).fetchone()[0]
                        if found!=len(cl): raise ValueError('Unknown target ID')
                    mw.writerow(m);cw.writerow(c);count+=1;pairs+=len(cl);matches+=len(ml);empty+=not ml;countries[country]=countries.get(country,0)+1
                if next(mr,None) is not None or next(cr,None) is not None: raise ValueError('Extra S1 rows')
            expected_start+=shard['count']
    records.close()
    if count!=run['queries']: raise ValueError('Wrong output count')
    mt.replace(mpath);ct.replace(cpath)
    identity={'run_sha256':sha256(run_dir/'run.json'),'rows':count,'partial':allow_partial}
    seal(mpath,identity);seal(cpath,identity)
    result={**identity,'pairs':pairs,'matches':matches,'empty_predictions':empty,'countries':countries,
            'seconds':time.monotonic()-started,'files':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in [mpath,cpath]},
            'submission_ready':False,'note':'Official full-test validator must still PASS. Partial outputs are not submissions.'}
    save_json(output/'assembly.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,default=ROOT/'artifacts/test_index')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    print(json.dumps(assemble(a.index,a.run_dir,a.output_dir,a.allow_partial),indent=2))
