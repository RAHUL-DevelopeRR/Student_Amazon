# Reproducible baseline: phases 1-4

Python 3.12 tested. Run from the repository root. No external business data or
services are used. Source TSVs and the official validator are read-only inputs.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r code/business_entity_resolution/requirements.txt
.venv/Scripts/python.exe -m unittest discover -s code/business_entity_resolution/src -p 'test_*.py' -v
.venv/Scripts/python.exe -u code/business_entity_resolution/src/profile_data.py
.venv/Scripts/python.exe -u code/business_entity_resolution/src/blocking.py --queries 1000 --distractors 30000
```

If Python 3.12 is not installed, use an available compatible interpreter to create
the venv. This workspace used the bundled Codex Python 3.12 runtime; run commands
after environment creation are independent of that absolute runtime path.

The audit creates `artifacts/audit.duckdb`, `artifacts/data_profile.json` and
`artifacts/data_profile.md`. It checkpoints per source file and verifies SHA-256
before reusing a completed source profile. Run only one database writer at a time.
Raw fields remain available alongside normalized fields. All imported columns are
VARCHAR to preserve IDs, leading zeros and missing-value semantics. The DuckDB
buffer limit is 512 MB with one worker and disk spill, not an absolute RSS bound.
Allow several GB of free disk for database and temporary files.
If normalization code changes, use a new `--artifacts` directory to rebuild the
audit; source fingerprints alone do not invalidate changed normalization logic.

The initial blocker is a **development pilot**, not validation of a final matcher:
1. Deterministically sample S1 IDs with seed 42.
2. Build a small search corpus from their true targets plus randomly sampled
   distractors. Labels only construct this artificial corpus and evaluate results;
   they never insert positives into returned candidates.
3. Fit separate name/address character 3-5 gram TF-IDF on a deterministic corpus
   sample. Sampled vocabulary/IDF is a measured resource compromise.
4. Scan targets in batches. Retain global top K independently per field and target
   source; union exact-name and rare-name-token blocks. Country compatibility is
   applied before top K and permits missing labels. All country strings supported.
5. Report recall, complete coverage, candidate counts, reduction ratio and the
   macro-F0.5 ceiling with a perfect matcher. Export missed pairs for diagnosis.

Exact-name blocks over 100 records and tokens over 30 records are skipped entirely,
not truncated to arbitrary first IDs. Their counts are reported. Base normalization
retains accents; accent folding is a separate utility. Expansions only use examples
provided in the problem statement; ambiguous `st` remains available in base text.

For a meaningful retrieval stress test, keep **all target distractors**:

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/blocking.py --queries 1000 --full-corpus --output-dir artifacts/blocking_full
```

For a fast full-corpus control without fuzzy retrieval:

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/blocking.py --queries 10000 --full-corpus --exact-only --output-dir artifacts/blocking_exact_full
```

This diagnostic unions exact names and exact addresses, skipping blocks larger
than 100 targets. Its recall is a lexical control, not the combined blocker's recall.

This is still a sampled-query training retrieval assessment, not held-out classifier
validation or a France score. Current retrieval bounds RAM, but scans/retransforms
the corpus per query batch; full S1 production requires cached indexes and more
compute. Do not mistake bounded memory for adequate full-production throughput.

`experiments/experiment_log.csv` records runs. Generated outputs/caches are ignored
by Git. The learned matcher pilot is available:

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/train.py --candidate-dir artifacts/blocking_pilot --output-dir artifacts/matcher_pilot
```

It builds 27 pair features and fits a fixed 180-tree LightGBM model. Normalized
name/address/country identities are hashed into training (60%), calibration (20%)
and evaluation (20%). Calibration alone chooses the decision threshold. IDs and
country identity are excluded from model features. All retrieved negative pairs
are retained. Missed retrieval positives count as false negatives in evaluation.
Model, metadata, split assignments and evaluation errors are saved locally.
The reduced corpus is easier than production: these are pilot scores, not
leaderboard estimates. Production test inference and output validation remain.

For a faster full-corpus token baseline (no injected true targets):

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/token_blocking.py --queries 3000
.venv/Scripts/python.exe -u code/business_entity_resolution/src/train.py --candidate-dir artifacts/blocking_token_full --output-dir artifacts/matcher_token_full
```

This scans every training target, then ranks candidates by summed IDF of shared
name/address tokens with target frequency at most 2000. Top 30 per source and field
are unioned with exact blocks. It is a lexical baseline: misspellings and cross-script
names can still prevent retrieval. Memory is bounded using DuckDB disk spill.
Query count is sampled; the target population is complete. Do not call this
full-production inference or a France evaluation.

Future final outputs must pass:

```powershell
.venv/Scripts/python.exe -X utf8 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Treat matches outside the exact scored candidate set as a pipeline failure even
though the official helper only warns. It skips ID existence by default.
