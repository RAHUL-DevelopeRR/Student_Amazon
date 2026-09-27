"""Measured, deadline-bounded challenger inference using the existing runner."""
import argparse,datetime,shutil
from pathlib import Path
import psutil
from config import ROOT,save_json
from inference_common import read_json,sha256
from predict import run
from predict_parallel import execute
from assemble_submission import assemble


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args()
    index=ROOT/'artifacts/test_index';baseline=ROOT/'artifacts/baseline-v2';pilot=ROOT/'artifacts/challenger_cloud_pilot'
    validation=read_json(ROOT/'artifacts/challenger/fresh_validation.json')
    config=read_json(baseline/'model_config.json');model=baseline/'model.txt'
    if sha256(model)!=validation['model_sha256'] or config['threshold']!=validation['threshold']:raise ValueError('Model differs from evaluated model')
    if validation['challenger']['macro_f05']<validation['baseline']['macro_f05']+.01 or validation['challenger']['pair_precision']<.94:raise ValueError('No sufficient fresh improvement')
    identity={n:sha256(Path(__file__).with_name(n)) for n in ['run_challenger.py','predict.py','predict_parallel.py','build_test_candidates.py','features.py','challenger_features.py','probe_joint_retrieval.py','normalize.py','postprocess.py']}
    identity['model']=sha256(baseline/'manifest.json')
    if not a.full:
        if pilot.exists():raise ValueError('Fresh pilot directory required')
        for name,digest in read_json(index/'manifest.json')['files'].items():
            if sha256(index/name)!=digest:raise ValueError('Corrupt index')
        report=execute(index,baseline,pilot,10000,4)
        serial=run(index,baseline,ROOT/'artifacts/challenger_serial_check',limit=250)
        for key in ['matching','scored']:
            if sha256(serial['shards'][0][key])!=sha256(report['shards'][0][key]):raise ValueError('Serial/parallel mismatch')
        save_json(pilot/'verified.json',{'identity':identity,'serial_parallel_parity':True,'seconds':report['this_invocation_seconds'],'pairs':report['pairs'],'output_bytes':report['output_bytes']})
    else:
        check=read_json(pilot/'verified.json')
        if check['identity']!=identity or not check['serial_parallel_parity']:raise ValueError('Pilot code/model changed')
        total=read_json(index/'manifest.json')['queries'];projected=check['seconds']*total/10000
        remaining=(datetime.datetime.fromisoformat('2026-09-27T18:29:00+00:00')-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        disk=shutil.disk_usage(ROOT).free;memory=psutil.virtual_memory().available
        gate={'projected_seconds':projected,'remaining_seconds':remaining,'runtime_pass':1.25*projected+2700<remaining,'disk_pass':check['output_bytes']*total/10000*2.8<disk*.85,'memory_pass':check['pairs']/10000*total*100<memory*.75,'identity':identity,'reserve_note':'25% runtime headroom plus 45 minutes; previous full assembly measured 760 seconds. Existing submitted baseline retained.'}
        save_json(ROOT/'artifacts/challenger_cloud_gate.json',gate)
        if not all(gate[k] for k in ['runtime_pass','disk_pass','memory_pass']):raise ValueError('Challenger feasibility gate failed')
        execute(index,baseline,ROOT/'artifacts/challenger_cloud_full',total,4)
        assemble(index,ROOT/'artifacts/challenger_cloud_full',ROOT/'output_v2')


if __name__=='__main__':main()
