"""Development-only recall probe using the complete target postings index."""
import csv,math,time
import numpy as np
from config import ROOT,save_json
from build_test_candidates import Retriever
from metric import read_mapping,entity_f05


def joint(r,q,k=60):
    chunks=[];weights=[]
    for field,text in enumerate(q[1:3]):
        for token in sorted(set(text.split())):
            if len(token)<3:continue
            p=r.cached(field,0,token)
            if p:
                df,ids=p;ids=ids[r.compatible(ids,q[3])]
                if len(ids):chunks.append(ids);weights.append(np.full(len(ids),math.log1p(r.meta['targets']/df)))
    if not chunks:return []
    ids,inv=np.unique(np.concatenate(chunks),return_inverse=True);scores=np.bincount(inv,weights=np.concatenate(weights));out=[]
    for mask in [ids<=r.meta['source2_max_tid'],ids>r.meta['source2_max_tid']]:
        pos=np.flatnonzero(mask);out.extend(map(int,ids[pos[np.lexsort((ids[pos],-scores[pos]))[:k]]]))
    return out


def main():
    start=time.monotonic();r=Retriever(ROOT/'artifacts/train_inference_index')
    base=read_mapping(ROOT/'artifacts/blocking_token_wide/candidate_pairs.tsv','candidate_entity_ids');truth={}
    with (ROOT/'dataset/train/train_ground_truth.tsv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['source1_entity_id'] in base:truth[row['source1_entity_id']]=set(filter(None,row['matched_entity_ids'].split(',')))
    found=total=pairs=0;ceil=[];out=ROOT/'artifacts/challenger';out.mkdir(exist_ok=True)
    with (out/'joint_candidates.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','candidate_entity_ids'])
        for j,q in enumerate(r.query_rows(0,3000)):
            targets=r.target_rows(joint(r,q));c=base[q[0]]|{t[0] for t in targets.values()};tr=truth[q[0]]
            found+=len(c&tr);total+=len(tr);pairs+=len(c);ceil.append(entity_f05(tr,c&tr));w.writerow([q[0],','.join(sorted(c))])
            if j%250==0:print('joint',j,flush=True)
    report={'scope':'development_3000_full_target_corpus','targets':r.meta['targets'],'candidate_recall':found/total,'oracle_macro_f05_ceiling':sum(ceil)/len(ceil),'pairs':pairs,'seconds':time.monotonic()-start,'seed':42,'joint_top_k_per_source':60}
    save_json(out/'joint_retrieval.json',report);print(report,flush=True);r.close()


if __name__=='__main__':main()
