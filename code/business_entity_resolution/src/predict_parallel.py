"""Run the unchanged scorer in disjoint processes; require a fresh cloud pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import platform
import time

from config import ROOT, save_json
from inference_common import read_json, sha256
from predict import run
from assemble_submission import assemble
from submission_gate import assess


def ranges(limit, workers):
    if limit <= 0 or not 1 <= workers <= min(limit, os.cpu_count() or 1):
        raise ValueError('Invalid query count or worker count')
    size = (limit + workers - 1) // workers
    return [(start, min(size, limit-start)) for start in range(0, limit, size)]


def execute(index, baseline, output, limit, workers):
    parts = ranges(limit, workers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {'workers': workers, 'host': platform.node(),
                   'runner_sha256': sha256(__file__)}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, index, baseline, output/f'part-{start}', start, count)
                   for start, count in parts]
        results = [future.result() for future in futures]
    elapsed = time.monotonic()-started
    result = dict(results[0])
    result['identity'] = dict(result['identity'], start=0, limit=limit)
    result['shards'] = [s for r in results for s in r['shards']]
    result.update(queries=limit, pairs=sum(r['pairs'] for r in results),
                  output_bytes=sum(r['output_bytes'] for r in results),
                  this_invocation_seconds=elapsed, parallel=fingerprint)
    result['average_candidates'] = result['pairs']/limit
    for key in list(result):
        if key.endswith('_seconds') and key not in ('projected_full_seconds', 'this_invocation_seconds'):
            result[key] = sum(r[key] for r in results)
    result['projected_full_seconds'] = elapsed*read_json(Path(index)/'manifest.json')['queries']/limit
    result['projection_note'] = 'Fresh parallel pilot wall time; same host and worker count required. Full resumed invocation is not a benchmark.'
    result.pop('candidate_quantiles_p50_p90_p99', None)
    result.pop('sampled_memory_max_bytes', None)
    save_json(output/'run.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, default=ROOT/'artifacts/test_index')
    p.add_argument('--baseline', type=Path, default=ROOT/'artifacts/baseline-v1')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=10000)
    p.add_argument('--pilot-dir', type=Path)
    a = p.parse_args()
    if a.limit != 10000:
        if a.limit != read_json(a.index/'manifest.json')['queries'] or not a.pilot_dir:
            p.error('Use exactly 10000 for the pilot, or the entire population with --pilot-dir')
        pilot = read_json(a.pilot_dir/'run.json')
        expected = {'workers': a.workers, 'host': platform.node(), 'runner_sha256': sha256(__file__)}
        if pilot.get('parallel') != expected or not pilot.get('baseline_output_parity'):
            raise ValueError('A matching host/worker pilot with exact output parity is required')
        if pilot['identity']['baseline_sha256'] != sha256(a.baseline/'manifest.json'):
            raise ValueError('Baseline changed since cloud pilot')
        gate = assess(a.pilot_dir/'run.json', ROOT/'artifacts/frozen_validation/validation.json', a.index)
        save_json(a.output_dir/'gate.json', gate)
        if not gate['full_run_allowed']:
            raise ValueError('Full-run feasibility gate failed')
    elif a.output_dir.exists():
        raise ValueError('Pilot requires a fresh output directory: cached work invalidates throughput')
    for name, digest in read_json(a.index/'manifest.json')['files'].items():
        if sha256(a.index/name) != digest:
            raise ValueError('Index checksum mismatch: '+name)
    result = execute(a.index, a.baseline, a.output_dir, a.limit, a.workers)
    if a.limit == 10000:
        actual = assemble(a.index, a.output_dir, a.output_dir/'assembled', allow_partial=True)
        expected = read_json(ROOT/'artifacts/test_pilot_10k/assembled/assembly.json')
        if any(actual['files'][name]['sha256'] != value['sha256'] for name, value in expected['files'].items()):
            raise ValueError('Cloud pilot differs from frozen laptop pilot')
        result['baseline_output_parity'] = True
        save_json(a.output_dir/'run.json', result)
    print(f"Completed {result['queries']:,} queries in {result['this_invocation_seconds']:.1f}s; projection {result['projected_full_seconds']/3600:.2f}h", flush=True)


if __name__ == '__main__':
    main()
