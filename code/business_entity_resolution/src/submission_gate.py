"""Conservative, explicit preflight gate; does not start jobs or tune models."""
import argparse
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
from config import ROOT,save_json
from inference_common import read_json,sha256,code_hash


def assess(pilot,validation,index,deadline='2026-09-27T18:29:00+00:00'):
    p,v,m=map(read_json,[pilot,validation,Path(index)/'manifest.json'])
    current=code_hash(['build_test_candidates.py','predict.py','postprocess.py','features.py','normalize.py'])
    if p['status']!='complete' or p['queries']!=10000 or p['identity']['start']!=0:raise ValueError('A completed exact 10K pilot is required')
    if p['identity']['index_sha256']!=sha256(Path(index)/'manifest.json'):raise ValueError('Pilot index mismatch')
    if p['identity']['baseline_sha256']!=v['baseline_sha256']:raise ValueError('Pilot/validation baseline mismatch')
    if p['identity']['code']!=current or v['inference_code']!=current:raise ValueError('Inference code changed after pilot or validation')
    remaining=(datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds()
    # Fixed before reading the untouched result; first-submission acceptance,
    # not a claim that these scores can win the challenge.
    quality=v['macro_f05']>=.82 and v['candidate_recall']>=.88 and v['pair_precision']>=.92
    projected=p['projected_full_seconds']
    needed_disk=p['output_bytes']*m['queries']/p['queries']*2.8
    free=shutil.disk_usage(ROOT).free
    # The official unmodified validator creates a new str and set entry for
    # every occurrence, plus per-query sets/mappings and all valid target IDs.
    string_bytes=sys.getsizeof('S2-123456789')
    validator_lower_bound=p['average_candidates']*m['queries']*(string_bytes+16)
    try:
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
        mem=Memory();mem.length=ctypes.sizeof(mem)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise OSError('memory query failed')
        available=mem.available
    except (AttributeError,OSError):available=None
    result={'quality_gate_pass':quality,'minimums':{'macro_f05':.82,'candidate_recall':.88,'pair_precision':.92},
            'remaining_seconds':remaining,'projected_inference_seconds':projected,
            'runtime_gate_pass':1.75*projected+7200<remaining,
            'projected_additional_disk_bytes':int(needed_disk),'free_disk_bytes':free,'disk_gate_pass':needed_disk<free*.85,
            'official_validator_candidate_memory_lower_bound_bytes':int(validator_lower_bound),
            'available_physical_memory_bytes':available,
            'validator_memory_gate_pass':available is not None and validator_lower_bound<available*.7,
            'deadline_utc':deadline,'note':'Runtime includes 75% headroom plus 2h for assembly/validation. Validator bound excludes other large mappings, so real memory is greater.'}
    result['full_run_allowed']=all(result[k] for k in ['quality_gate_pass','runtime_gate_pass','disk_gate_pass','validator_memory_gate_pass'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=ROOT/'artifacts/test_pilot_10k/run.json')
    p.add_argument('--validation',type=Path,default=ROOT/'artifacts/frozen_validation/validation.json')
    p.add_argument('--index',type=Path,default=ROOT/'artifacts/test_index')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/submission_gate.json');a=p.parse_args()
    result=assess(a.pilot,a.validation,a.index);save_json(a.output,result);print(json.dumps(result,indent=2))
