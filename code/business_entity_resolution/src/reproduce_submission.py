"""Replay the packaged frozen model; no training or threshold selection."""
import argparse
from pathlib import Path
from config import ROOT
from inference_common import read_json
from predict_parallel import execute
from assemble_submission import assemble
from validate_outputs import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT/'code/business_entity_resolution/model')
    p.add_argument('--index', type=Path, default=ROOT/'artifacts/test_index')
    p.add_argument('--run-dir', type=Path, default=ROOT/'artifacts/reproduced_full')
    p.add_argument('--output-dir', type=Path, default=ROOT/'reproduced_output')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    total = read_json(a.index/'manifest.json')['queries']
    if a.output_dir.resolve() == (ROOT/'output').resolve():
        p.error('Preserve packaged outputs; choose a separate output directory')
    execute(a.index, a.model, a.run_dir, total, a.workers)
    assemble(a.index, a.run_dir, a.output_dir)
    if not validate(a.output_dir, ROOT/'dataset/test')['submission_ready']:
        raise SystemExit('Official validation failed')


if __name__ == '__main__':
    main()
