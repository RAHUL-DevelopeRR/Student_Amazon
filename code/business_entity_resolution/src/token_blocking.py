"""Full-target token retrieval using disk-backed SQL, no ground-truth injection."""
import argparse,csv,json,time
from pathlib import Path
from config import ROOT,connect,save_json,log_experiment
from blocking import metrics,exact_diagnostic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queries',type=int,default=3000)
    p.add_argument('--top-k',type=int,default=30)
    p.add_argument('--max-df',type=int,default=2000)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/blocking_token_full')
    p.add_argument('--db',type=Path,default=ROOT/'artifacts/audit.duckdb')
    a=p.parse_args()
    if min(a.queries,a.top_k,a.max_df)<1:p.error('Counts must be positive')
    start=time.monotonic();con=connect(a.db)
    con.execute(f"CREATE TEMP TABLE queries AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM train_source1 ORDER BY md5(entity_id || '42') LIMIT {a.queries}")
    queries=con.execute('SELECT * FROM queries ORDER BY entity_id').fetchall()
    con.execute("CREATE TEMP VIEW corpus AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM targets")
    count=con.execute('SELECT count(*) FROM corpus').fetchone()[0]
    truth={i:set(v.split(',')) if v else set() for i,v in con.execute('SELECT source1_entity_id,matched_entity_ids FROM truth JOIN queries ON source1_entity_id=entity_id').fetchall()}
    candidate,channels,caps=exact_diagnostic(con,queries)
    lookup={q[0]:i for i,q in enumerate(queries)}
    for field in ['name_norm','address_norm']:
        print(f'{field}: scanning full target population for query tokens',flush=True)
        con.execute(f"CREATE OR REPLACE TEMP TABLE query_tokens AS SELECT DISTINCT entity_id,country,unnest(string_split({field},' ')) token FROM queries")
        con.execute("DELETE FROM query_tokens WHERE length(token)<3")
        con.execute(f"CREATE OR REPLACE TEMP TABLE postings AS SELECT DISTINCT entity_id,country,token FROM (SELECT entity_id,country,unnest(string_split({field},' ')) token FROM corpus) t SEMI JOIN query_tokens q USING(token)")
        con.execute('CREATE OR REPLACE TEMP TABLE token_df AS SELECT token,count(*) df FROM postings GROUP BY token')
        con.execute(f"CREATE OR REPLACE TEMP TABLE scored AS SELECT q.entity_id qid,t.entity_id tid,sum(ln(1.0+{count}/d.df)) evidence FROM query_tokens q JOIN token_df d USING(token) JOIN postings t USING(token) WHERE d.df<={a.max_df} AND (q.country=t.country OR q.country='' OR t.country='') GROUP BY q.entity_id,t.entity_id")
        results=con.execute(f"SELECT qid,tid FROM scored QUALIFY row_number() OVER(PARTITION BY qid,left(tid,2) ORDER BY evidence DESC,tid)<={a.top_k}").fetchall()
        channel=[set() for _ in queries]
        for q,t in results:channel[lookup[q]].add(t)
        channels[field+'_token']=channel
        for i,c in enumerate(channel):candidate[i].update(c)
        print(f'{field}: retained {len(results):,} candidate pairs',flush=True)
    result=metrics(queries,truth,candidate,count)
    result.update(scope='sampled_queries_full_target_corpus_token_retrieval',seconds=time.monotonic()-start,
                  parameters={'queries':a.queries,'top_k':a.top_k,'max_df':a.max_df,'seed':42},
                  channel_metrics={k:metrics(queries,truth,v,count) for k,v in channels.items()},caps=caps)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    save_json(a.output_dir/'metrics.json',result)
    with (a.output_dir/'candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id','candidate_entity_ids'])
        w.writerows((q[0],','.join(sorted(c))) for q,c in zip(queries,candidate))
    log_experiment('token_blocking',{k:result[k] for k in ['scope','candidate_recall','average_candidates','seconds']},result['parameters'])
    print(json.dumps({k:v for k,v in result.items() if k!='channel_metrics'},indent=2),flush=True)
    con.close()


if __name__=='__main__':main()
