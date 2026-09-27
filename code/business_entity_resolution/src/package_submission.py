"""Package only hash-verified, officially validated outputs and a frozen model."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from config import ROOT
from inference_common import read_json, sha256


def package(output, model, methodology, destination):
    report = read_json(output/'validation.json')
    if not report.get('submission_ready') or report.get('pilot') or not report.get('pass'):
        raise ValueError('A passing full official validation report is required')
    if report['validator_sha256'] != sha256(ROOT/'utils/validate_submission.py'):
        raise ValueError('Official validator fingerprint mismatch')
    manifest = read_json(model/'manifest.json')
    for name, digest in manifest['identity']['files'].items():
        if sha256(model/name) != digest:
            raise ValueError('Frozen model changed: '+name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.zip.partial')
    source = ROOT/'code/business_entity_resolution'
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for name in ['matching_results.tsv', 'candidate_pairs.tsv']:
            h = hashlib.sha256(); size = 0
            with (output/name).open('rb') as src, z.open('output/'+name, 'w', force_zip64=True) as dst:
                for block in iter(lambda: src.read(8*1024*1024), b''):
                    dst.write(block); h.update(block); size += len(block)
            if {'sha256': h.hexdigest(), 'bytes': size} != report['files'][name]:
                raise ValueError('Output differs from official validated file: '+name)
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc' and 'model' not in path.relative_to(source).parts:
                z.write(path, path.relative_to(ROOT).as_posix())
        for name in [*manifest['identity']['files'], 'manifest.json']:
            z.write(model/name, 'code/business_entity_resolution/model/'+name)
        z.write(methodology, 'Documentation_template.md')
        z.write(ROOT/'utils/validate_submission.py', 'utils/validate_submission.py')
        for name in ['validation.json', 'assembly.json', 'official_validator.log']:
            z.write(output/name, 'verification/'+name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    temp.replace(destination)
    receipt = {'zip': str(destination.resolve()), 'bytes': destination.stat().st_size,
               'sha256': sha256(destination), 'model_sha256': sha256(model/'model.txt'),
               'files': report['files'], 'crc_verified': True}
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--methodology', type=Path, default=ROOT/'Documentation_template.md')
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    package(a.output_dir, a.model, a.methodology, a.destination)
