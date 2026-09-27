import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from build_test_candidates import build,Retriever,candidate_shard
from inference_common import freeze,sha256,valid
from config import ROOT
from predict import run
from assemble_submission import assemble


class InferenceTests(unittest.TestCase):
    def test_topk_global_frequency_country_and_oversized_exact_block(self):
        from collections import Counter
        from math import log1p
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            values={1:[['S1-a','shared name','12 avenue','France']],
                    2:[[f'S2-{i:03d}','shared name',f'{i} avenue','US' if i<5 else 'France'] for i in range(120)],
                    3:[[f'S3-{i:03d}','shared name',f'{i} avenue',''] for i in range(65)]}
            for s,rows in values.items():
                with (root/f'test_source{s}.tsv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country']);w.writerows(rows)
            build(root,root/'index');r=Retriever(root/'index');q=('S1-a','shared name','12 avenue','France')
            raw=values[2]+values[3];expected=set()
            for field in [1,2]:
                df=Counter(t for row in raw for t in set(row[field].split()) if len(t)>=3)
                exact=Counter(row[field] for row in raw)
                if q[field] and exact[q[field]]<=100:
                    expected.update(row[0] for row in raw if row[field]==q[field] and (not row[3] or row[3].lower()==q[3].lower()))
                for prefix in ['S2-','S3-']:
                    scored=[]
                    for row in raw:
                        if not row[0].startswith(prefix) or (row[3] and row[3]!=q[3]):continue
                        score=sum(log1p(len(raw)/df[t]) for t in sorted(set(q[field].split())&set(row[field].split())) if len(t)>=3 and df[t]<=20000)
                        if score:scored.append((-score,row[0]))
                    expected.update(i for _,i in sorted(scored)[:60])
            found={v[0] for v in r.target_rows(r.retrieve(q)).values()};r.close()
            self.assertEqual(found,expected)
            self.assertNotIn('S2-000',found)
            self.assertNotIn('S2-119',found)

    def test_resume_country_ties_and_scored_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir()
            rows={1:[['S1-a','Alpha Bakery','12 road','France'],['S1-b','','','Japan']],
                  2:[['S2-a','Alpha Bakery','12 road','France'],['S2-b','Alpha Bakery','12 road','US']],
                  3:[['S3-a','Alpha Bakery','12 road',''],['S3-b','Other','99','Japan']]}
            for s,values in rows.items():
                with (data/f'test_source{s}.tsv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country']);w.writerows(values)
            index=root/'index';meta=build(data,index)
            self.assertEqual(meta['targets'],4)
            r=Retriever(index)
            found=r.retrieve(('S1-a','alpha bakery','12 road','France'))
            self.assertEqual({v[0] for v in r.target_rows(found).values()},{'S2-a','S3-a'})
            self.assertEqual(r.retrieve(('S1-b','','','Japan')),[])
            shard=candidate_shard(r,root/'c',0,2);h=sha256(shard);mtime=shard.stat().st_mtime_ns
            self.assertEqual(candidate_shard(r,root/'c',0,2),shard);self.assertEqual(shard.stat().st_mtime_ns,mtime)
            shard.write_text('broken');self.assertFalse(valid(shard,{}))
            candidate_shard(r,root/'c',0,2);self.assertEqual(sha256(shard),h);r.close()
            # Use a tiny fixture model, independent of generated project artifacts.
            import lightgbm as lgb
            from features import pair_features
            source=root/'model';source.mkdir()
            x=np.array([pair_features(('S1','alpha','12',''),('S2-a','alpha' if i%2 else 'other','12','')) for i in range(40)])
            model=lgb.LGBMClassifier(n_estimators=2,min_child_samples=1,num_leaves=2,verbosity=-1,n_jobs=1).fit(x,[i%2 for i in range(40)])
            model.booster_.save_model(str(source/'model.txt'))
            for name,value in [('model_config.json',{'threshold':.48,'feature_count':27}),('metrics.json',{}),('splits.json',{})]: (source/name).write_text(json.dumps(value))
            baseline=root/'baseline';freeze(source,baseline)
            result=run(index,baseline,root/'run',limit=2,shard_size=1)
            self.assertEqual(result['queries'],2);self.assertEqual(result['pairs'],2)
            self.assertEqual(run(index,baseline,root/'run',limit=2,shard_size=1)['pairs'],2)
            a=assemble(index,root/'run',root/'output')
            self.assertEqual(a['rows'],2);self.assertEqual(a['countries'],{'Japan':1,'France':1})
            from predict_parallel import execute, ranges
            self.assertEqual(ranges(5,2),[(0,3),(3,2)])
            execute(index,baseline,root/'parallel',2,2)
            parallel=assemble(index,root/'parallel',root/'parallel_output')
            self.assertEqual(parallel['files'],a['files'])
            import subprocess,sys
            check=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'utils/validate_submission.py'),'--matching',str(root/'output/matching_results.tsv'),'--candidate',str(root/'output/candidate_pairs.tsv'),'--test-dir',str(data),'--check-ids'],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(check.returncode,0,check.stdout+check.stderr)
            from validate_outputs import validate
            assemble(index,root/'run',root/'pilot_output',allow_partial=True)
            checked=validate(root/'pilot_output',data,pilot=True)
            self.assertTrue(checked['pass'])
            self.assertFalse(checked['submission_ready'])


if __name__=='__main__':unittest.main()
