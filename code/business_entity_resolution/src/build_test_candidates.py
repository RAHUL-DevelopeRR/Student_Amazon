"""Build reusable lexical indexes from three source TSVs, then retrieve shards.

Default mode reads test sources only. Explicit --split train builds an independent
validation index; this module never opens ground truth. No candidate pruning.
"""
import argparse
import csv
import json
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from config import ROOT,connect,save_json
from normalize import normalize
from inference_common import sha256,code_hash,read_json,sqlite,seal,valid,sampled_rss

POLICY={'max_df':20000,'top_k':60,'exact_cap':100,'seed':42,'version':1}


def build(data,index,split='test'):
    data,index=Path(data),Path(index);index.mkdir(parents=True,exist_ok=True)
    sources=[data/f'{split}_source{i}.tsv' for i in range(1,4)]
    signature={'sources':{p.name:sha256(p) for p in sources},'policy':POLICY,
               'normalization':code_hash(['normalize.py']),'format_version':1,'split':split}
    statepath=index/'build_state.json'
    state=read_json(statepath) if statepath.exists() else {'signature':signature,'done':[],'seconds':0}
    if state['signature']!=signature: raise ValueError('Index inputs changed; use a fresh index directory')
    if (index/'manifest.json').exists():
        m=read_json(index/'manifest.json')
        for name,h in m['files'].items():
            if sha256(index/name)!=h: raise ValueError('Index checksum mismatch: '+name)
        return m
    save_json(statepath,state)
    db=connect(index/'build.duckdb',memory='1GB')
    def stage(name,fn):
        if name in state['done']: return
        print('BUILD '+name,flush=True);start=time.monotonic()
        fn();db.execute('CHECKPOINT')
        state['done'].append(name);state['seconds']+=time.monotonic()-start
        save_json(statepath,state)
    for number,path in enumerate(sources,1):
        def import_file(number=number,path=path):
            tmp=index/'normalized.tmp.tsv'
            with path.open(encoding='utf-8-sig',newline='') as src,tmp.open('w',encoding='utf-8',newline='') as dst:
                r=csv.reader(src,delimiter='\t');w=csv.writer(dst,delimiter='\t',lineterminator='\n')
                if next(r)!=['entity_id','business_name','business_address','country']: raise ValueError('Bad source schema')
                w.writerow(['entity_id','name','address','country'])
                for i,row in enumerate(r,1):
                    if len(row)!=4 or not row[0].startswith(f'S{number}-'): raise ValueError('Bad source row')
                    w.writerow([row[0],normalize(row[1]),normalize(row[2]),row[3]])
                    if i%1000000==0: print(f'  source{number}: {i:,}',flush=True)
            db.execute(f"CREATE OR REPLACE TABLE s{number} AS SELECT * FROM read_csv(?,delim='\t',header=true,all_varchar=true,nullstr='')",[str(tmp)])
            db.execute(f"UPDATE s{number} SET name=coalesce(name,''),address=coalesce(address,''),country=coalesce(country,'')")
            if db.execute(f'SELECT count(*)-count(DISTINCT entity_id) FROM s{number}').fetchone()[0]: raise ValueError('Duplicate source IDs')
            tmp.unlink()
        stage(f'source{number}',import_file)
    def records():
        offset=0
        for source in [2,3]:
            # Sort only narrow IDs, not millions of name/address strings in a window.
            db.execute(f'CREATE OR REPLACE TABLE target_ids AS SELECT (row_number() OVER(ORDER BY entity_id)+{offset})::UINTEGER tid,entity_id FROM s{source}')
            select=f'SELECT i.tid,s.* FROM s{source} s JOIN target_ids i USING(entity_id)'
            db.execute(('CREATE OR REPLACE TABLE targets AS ' if source==2 else 'INSERT INTO targets ')+select)
            offset+=db.execute(f'SELECT count(*) FROM s{source}').fetchone()[0]
        db.execute('DROP TABLE target_ids')
        db.execute("CREATE OR REPLACE TABLE queries AS SELECT row_number() OVER(ORDER BY md5(entity_id || '42'),entity_id)::INTEGER ordinal,* FROM s1")
        out=sqlite(index/'records.sqlite')
        out.executescript('DROP TABLE IF EXISTS targets; DROP TABLE IF EXISTS queries; CREATE TABLE targets(tid INTEGER PRIMARY KEY,entity_id TEXT,name TEXT,address TEXT,country TEXT); CREATE TABLE queries(ordinal INTEGER PRIMARY KEY,entity_id TEXT,name TEXT,address TEXT,country TEXT);')
        for table in ['targets','queries']:
            # External ORDER BY spills; unlike the earlier window over payloads,
            # it also avoids random-page SQLite insertion for millions of rows.
            cur=db.execute(f'SELECT * FROM {table} ORDER BY 1')
            while rows:=cur.fetchmany(10000): out.executemany(f'INSERT INTO {table} VALUES(?,?,?,?,?)',rows)
            out.commit()
        out.execute('CREATE UNIQUE INDEX target_entity_id ON targets(entity_id)')
        out.execute('CREATE UNIQUE INDEX query_entity_id ON queries(entity_id)')
        out.commit()
        out.close()
        countries=[r[0] for r in db.execute('SELECT DISTINCT country FROM targets ORDER BY country').fetchall()]
        codes={c:i for i,c in enumerate(countries)}
        n=db.execute('SELECT count(*) FROM targets').fetchone()[0]
        arr=np.lib.format.open_memmap(index/'countries.npy',mode='w+',dtype=np.uint32,shape=(n+1,))
        cur=db.execute('SELECT tid,country FROM targets')
        while rows:=cur.fetchmany(10000):
            for tid,c in rows: arr[tid]=codes[c]
        arr.flush();del arr
        save_json(index/'countries.json',countries)
    stage('records',records)
    out=sqlite(index/'postings.sqlite')
    out.execute('CREATE TABLE IF NOT EXISTS postings(field INTEGER,kind INTEGER,token TEXT,df INTEGER,ids BLOB,PRIMARY KEY(field,kind,token)) WITHOUT ROWID')
    for field,column in enumerate(['name','address']):
        def tokens(column=column):
            db.execute(f"CREATE OR REPLACE TABLE tokens AS SELECT tid,token,hash(token)%32 bucket FROM (SELECT tid,unnest(list_distinct(string_split({column},' '))) token FROM targets) WHERE length(token)>=3")
            db.execute('CREATE OR REPLACE TABLE df AS SELECT token,count(*) n FROM tokens GROUP BY token')
        stage(f'tokens{field}',tokens)
        for bucket in range(32):
            def export(bucket=bucket,field=field):
                cur=db.execute('SELECT t.token,d.n,list_sort(list(t.tid)) FROM tokens t JOIN df d USING(token) WHERE t.bucket=? AND d.n<=? GROUP BY t.token,d.n ORDER BY t.token',[bucket,POLICY['max_df']])
                while rows:=cur.fetchmany(500):
                    out.executemany('INSERT OR REPLACE INTO postings VALUES(?,?,?,?,?)',
                        [(field,0,t,n,np.asarray(ids,dtype='<u4').tobytes()) for t,n,ids in rows])
                out.commit()
            stage(f'postings{field}_{bucket}',export)
        for bucket in range(32):
            def exact(field=field,column=column,bucket=bucket):
                cur=db.execute(f"SELECT {column},count(*),list_sort(list(tid)) FROM targets WHERE {column}<>'' AND hash({column})%32=? GROUP BY {column} HAVING count(*)<=100 ORDER BY {column}",[bucket])
                while rows:=cur.fetchmany(500):
                    out.executemany('INSERT OR REPLACE INTO postings VALUES(?,?,?,?,?)',
                        [(field,1,t,n,np.asarray(ids,dtype='<u4').tobytes()) for t,n,ids in rows])
                out.commit()
            stage(f'exact{field}_{bucket}',exact)
    out.close()
    n=db.execute('SELECT count(*) FROM targets').fetchone()[0]
    q=db.execute('SELECT count(*) FROM queries').fetchone()[0]
    boundary=db.execute("SELECT max(tid) FROM targets WHERE starts_with(entity_id,'S2-')").fetchone()[0] or 0
    db.close()
    m={'signature':signature,'builder_sha256':sha256(Path(__file__)),'targets':n,'queries':q,'source2_max_tid':boundary,'build_seconds':state['seconds'],
       'files':{name:sha256(index/name) for name in ['records.sqlite','postings.sqlite','countries.npy','countries.json']}}
    save_json(index/'manifest.json',m)
    return m


class Retriever:
    def __init__(self,index):
        self.index=Path(index);self.meta=read_json(self.index/'manifest.json')
        self.records=sqlite(self.index/'records.sqlite',True)
        self.postings=sqlite(self.index/'postings.sqlite',True)
        self.countries=np.load(self.index/'countries.npy',mmap_mode='r')
        self.labels=read_json(self.index/'countries.json')
        self.cached=lru_cache(maxsize=512)(self.lookup)

    def lookup(self,field,kind,token):
        row=self.postings.execute('SELECT df,ids FROM postings WHERE field=? AND kind=? AND token=?',(field,kind,token)).fetchone()
        return None if row is None else (row[0],np.frombuffer(row[1],dtype='<u4'))

    def compatible(self,ids,country,exact=False):
        codes=[i for i,c in enumerate(self.labels) if not country or not c or (c.lower()==country.lower() if exact else c==country)]
        return np.isin(self.countries[ids],codes)

    def retrieve(self,q):
        selected=set();N=self.meta['targets'];boundary=self.meta['source2_max_tid']
        for field,text in enumerate(q[1:3]):
            exact=self.cached(field,1,text) if text else None
            if exact:
                ids=exact[1];selected.update(map(int,ids[self.compatible(ids,q[3],True)]))
            chunks=[];weights=[]
            for token in sorted(set(text.split())):
                if len(token)<3: continue
                posting=self.cached(field,0,token)
                if posting:
                    df,ids=posting;ids=ids[self.compatible(ids,q[3])]
                    if len(ids): chunks.append(ids);weights.append(np.full(len(ids),math.log1p(N/df)))
            if not chunks: continue
            ids,inverse=np.unique(np.concatenate(chunks),return_inverse=True)
            scores=np.bincount(inverse,weights=np.concatenate(weights))
            for source in [ids<=boundary,ids>boundary]:
                positions=np.flatnonzero(source)
                order=np.lexsort((ids[positions],-scores[positions]))[:POLICY['top_k']]
                selected.update(map(int,ids[positions[order]]))
        return sorted(selected)

    def query_rows(self,start,count):
        return self.records.execute('SELECT entity_id,name,address,country FROM queries WHERE ordinal>? AND ordinal<=? ORDER BY ordinal',(start,start+count)).fetchall()

    def target_rows(self,ids):
        result={}
        for start in range(0,len(ids),500):
            batch=ids[start:start+500]
            for row in self.records.execute('SELECT * FROM targets WHERE tid IN ('+','.join('?' for _ in batch)+')',batch): result[row[0]]=row[1:]
        return result

    def close(self):
        self.cached.cache_clear();self.postings.close();self.records.close()
        self.countries._mmap.close()


def candidate_shard(retriever,directory,start,count):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    dest=directory/f'{start:09d}.candidates.jsonl'
    identity={'index':sha256(retriever.index/'manifest.json'),'start':start,'count':count,
              'code':code_hash(['build_test_candidates.py','normalize.py'])}
    if valid(dest,identity): return dest
    rows=retriever.query_rows(start,count)
    if len(rows)!=count: raise ValueError('Query range outside index')
    begin=time.monotonic();counts=[];tmp=dest.with_suffix('.tmp');retrieval_seconds=output_seconds=0.0
    with tmp.open('w',encoding='utf-8') as f:
        for q in rows:
            started=time.monotonic();ids=retriever.retrieve(q);counts.append(len(ids))
            retrieval_seconds+=time.monotonic()-started;started=time.monotonic()
            f.write(json.dumps([q,ids],ensure_ascii=False)+'\n')
            output_seconds+=time.monotonic()-started
    tmp.replace(dest)
    seal(dest,dict(identity,seconds=time.monotonic()-begin,pairs=sum(counts),
                   counts=counts,retrieval_seconds=retrieval_seconds,output_seconds=output_seconds,sampled_rss=sampled_rss()))
    return dest


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'dataset/test')
    p.add_argument('--index',type=Path,default=ROOT/'artifacts/test_index')
    p.add_argument('--split',choices=['test','train'],default='test')
    a=p.parse_args();print(json.dumps(build(a.data_dir,a.index,a.split),indent=2))


if __name__=='__main__': main()
