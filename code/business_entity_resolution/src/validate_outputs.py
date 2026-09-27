"""Run the unchanged official validator and record its real exit/output.

Pilot mode creates an explicit S1 subset and hard-links the original target TSVs.
It never describes a partial output as a valid full submission.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from config import ROOT,save_json
from inference_common import read_json,sha256


def check_namespace(data,output):
    """Separate final audit; inference itself never opens training data."""
    from config import connect
    train=data.parent/'train'
    sources=[train/f'train_source{s}.tsv' for s in [1,2,3]]
    if not all(p.exists() for p in sources):raise ValueError('Training source IDs required for namespace audit')
    db=connect(output/'namespace_audit.duckdb')
    db.execute("CREATE OR REPLACE TABLE test_ids AS SELECT entity_id FROM read_csv(?,delim='\t',header=true,all_varchar=true)",[[str(data/f'test_source{s}.tsv') for s in [1,2,3]]])
    overlap=db.execute("SELECT count(*) FROM read_csv(?,delim='\t',header=true,all_varchar=true) t SEMI JOIN test_ids USING(entity_id)",[[str(p) for p in sources]]).fetchone()[0]
    db.close()
    report={'overlapping_train_test_ids':overlap,'source':'original source TSV IDs only; no ground truth used'}
    save_json(output/'namespace_audit.json',report)
    if overlap:raise ValueError('Train/test IDs overlap; cannot assert training-ID exclusion')
    return report


def validate(output,data,pilot=False):
    output,data=Path(output),Path(data)
    assembly=read_json(output/'assembly.json')
    if assembly['partial']!=pilot: raise ValueError('Pilot/full validation mode mismatch')
    for name,metadata in assembly['files'].items():
        if sha256(output/name)!=metadata['sha256']:raise ValueError('Assembled file changed before validation')
    if not pilot:check_namespace(data,output)
    validation_data=data
    if pilot:
        validation_data=output/'pilot_validation_inputs';validation_data.mkdir(exist_ok=True)
        with (output/'matching_results.tsv').open(encoding='utf-8',newline='') as f:
            wanted={r['source1_entity_id'] for r in csv.DictReader(f,delimiter='\t')}
        with (data/'test_source1.tsv').open(encoding='utf-8-sig',newline='') as src,(validation_data/'test_source1.tsv').open('w',encoding='utf-8',newline='') as dst:
            r=csv.reader(src,delimiter='\t');w=csv.writer(dst,delimiter='\t',lineterminator='\n');w.writerow(next(r))
            count=0
            for row in r:
                if row[0] in wanted:w.writerow(row);count+=1
        if count!=len(wanted):raise ValueError('Pilot IDs absent from original test input')
        for s in [2,3]:
            src=data/f'test_source{s}.tsv';dst=validation_data/src.name
            if dst.exists():
                if not os.path.samefile(src,dst):raise ValueError('Pilot target link points elsewhere')
            else:os.link(src,dst)
    command=[sys.executable,'-X','utf8',str(ROOT/'utils/validate_submission.py'),
             '--matching',str(output/'matching_results.tsv'),'--candidate',str(output/'candidate_pairs.tsv'),
             '--test-dir',str(validation_data),'--check-ids']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    (output/'official_validator.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    passed=result.returncode==0 and 'PASS' in result.stdout and 'WARNING:' not in result.stdout
    report={'exit_code':result.returncode,'pass':passed,'pilot':pilot,'command':command,
            'validator_sha256':sha256(ROOT/'utils/validate_submission.py'),'files':assembly['files'],
            'submission_ready':passed and not pilot}
    save_json(output/'validation.json',report)
    if passed and not pilot:
        assembly['submission_ready']=True;save_json(output/'assembly.json',assembly)
    print(result.stdout+result.stderr,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--data-dir',type=Path,default=ROOT/'dataset/test');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    result=validate(a.output_dir,a.data_dir,a.pilot)
    sys.exit(0 if result['pass'] else 1)
