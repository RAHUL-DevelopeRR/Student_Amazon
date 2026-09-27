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

For a wider lexical retrieval experiment on the same fixed queries:

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/token_blocking.py --queries 3000 --max-df 20000 --top-k 60 --memory-limit 1GB --output-dir artifacts/blocking_token_wide
.venv/Scripts/python.exe -u code/business_entity_resolution/src/train.py --candidate-dir artifacts/blocking_token_wide --output-dir artifacts/matcher_token_wide
```

The buffer limit is configurable because widening the token cutoff increases
intermediate join size. Completed field tables are dropped before the next field
is processed. It is not a process RSS limit. Each completed run includes country
metrics and `missed_links.tsv`; labels are consulted only for evaluation/export.
Pair features use an 8,192-entry preprocessing cache without changing their values.
Repeated experiments on these development queries require a fresh untouched
evaluation sample before reporting a final generalization score.

Future final outputs must pass:

```powershell
.venv/Scripts/python.exe -X utf8 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Treat matches outside the exact scored candidate set as a pipeline failure even
though the official helper only warns. It skips ID existence by default.

## Resumable frozen-model inference

Measured status (27 Sep): the 10K pilot passed the official validator, and fresh
1K training validation reached macro F0.5 0.872322. The full-run gate failed:
27.1 hours projected inference and at least 24.3 GiB of validator candidate RAM.
No complete submission was generated. See `experiments/production-inference-20260927.json`.

The production path does not import the audit/training code or read ground truth.
It builds a separate index from three explicitly selected source TSVs. The default
is test. SQLite stores normalized records and compact uint32 token postings;
NumPy aggregates shared-token IDF and retains top 60 per field/source. Exact blocks
retain the existing cap of 100. Global target token frequency remains capped at
20,000. Country filtering happens before top-k. No new candidate pruning, neural
model, retraining, or deterministic scoring bypass is enabled.

Freeze the existing model once (the CLI can do this when predicting):

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/build_test_candidates.py
.venv/Scripts/python.exe -u code/business_entity_resolution/src/predict.py --freeze-from artifacts/matcher_token_wide --limit 10000 --output-dir artifacts/test_pilot_10k
.venv/Scripts/python.exe code/business_entity_resolution/src/assemble_submission.py --run-dir artifacts/test_pilot_10k --output-dir artifacts/test_pilot_10k/assembled --allow-partial
.venv/Scripts/python.exe -X utf8 code/business_entity_resolution/src/validate_outputs.py --output-dir artifacts/test_pilot_10k/assembled --pilot
```

`baseline-v1` copies model, settings, metrics and split assignments with SHA-256
checksums and feature/dependency fingerprints. An existing frozen baseline cannot
be silently overwritten. Index imports checkpoint after each source; posting
exports checkpoint by field and bucket. A completed index is verified against its
input and artifact hashes on build reuse. Interrupted source imports restart that
source. Build scratch is retained for recovery and consumes additional disk.

S1 order is deterministic `md5(entity_id || '42'), entity_id`. Each 250-query
retrieval shard and each scored output has a checksum manifest. Files are written
to temporary paths and renamed only after completion; corrupt or partial shards
are regenerated. Run signatures reject changed model, policy or feature code.
Scoring loads at most 32 queries' candidate records/features at once. Empty and
multiple matches are retained. `candidate_pairs.tsv` is written from the exact
candidate IDs supplied to the LightGBM model, not from an earlier unscored pool.

The pilot's `run.json` contains separate retrieval, candidate loading, feature,
model and output timings, total time, sampled RSS, candidate quantiles, output
bytes and a linear full-run projection. Retrieval timing includes its JSONL shard
writing. Reused-shard timings are original computation times; invocation wall time
is separately recorded. System sleep and other workloads affect wall time.

Untouched validation uses an entirely separate training index and the next 1,000
queries after the 3,000 development IDs. It checks development ID consistency and
rejects overlapping normalized identities before inference. Labels are read only
by the explicit training validation command. Do not tune after its first result.

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/build_test_candidates.py --split train --data-dir dataset/train --index artifacts/train_inference_index
.venv/Scripts/python.exe -u code/business_entity_resolution/src/validate_frozen.py --cascade --output-dir artifacts/cascade_measurement
.venv/Scripts/python.exe -u code/business_entity_resolution/src/validate_frozen.py
.venv/Scripts/python.exe code/business_entity_resolution/src/submission_gate.py
```

The cascade command measures exact nonempty name AND address matches on development
queries, including precision and macro-F0.5 change, but keeps the bypass disabled.
It also compares indexed retrieval with the previous SQL candidates. The gate
requires macro-F0.5 >=0.82, recall >=0.88 and precision >=0.92, fixed before reading
fresh validation. Runtime needs 75% headroom plus two hours for assembly/validation.
Disk and official-validator memory must also fit. These are first-submission
feasibility checks, not a winning-score estimate. Do not start the full run if any
gate fails. The official validator retains all candidates in Python memory; this
can require substantially more RAM than inference itself.

Only after the gate passes:

```powershell
.venv/Scripts/python.exe -u code/business_entity_resolution/src/predict.py --limit 1732544 --output-dir artifacts/test_full
.venv/Scripts/python.exe code/business_entity_resolution/src/assemble_submission.py --run-dir artifacts/test_full
.venv/Scripts/python.exe -X utf8 code/business_entity_resolution/src/validate_outputs.py --output-dir output
```

The full assembler refuses partial coverage. It independently verifies S1 order,
each target's existence, duplicate-free lists, match containment and country counts.
Partial pilot files are never submission-ready. Assembly metadata remains marked
`submission_ready: false` until the complete official validation has been observed
to PASS. No script uploads files to Unstop.

## Cloud continuation (2026-09-27)

Use Python 3.12 and the pinned requirements. Transfer files byte-for-byte: Git
checkout line-ending conversion can invalidate the frozen code fingerprints.
Required inputs: code/business_entity_resolution, utils/validate_submission.py,
all six source TSVs under dataset/train and dataset/test, artifacts/baseline-v1,
artifacts/test_index/{manifest.json,records.sqlite,postings.sqlite,countries.npy,countries.json},
artifacts/frozen_validation/validation.json, and
artifacts/test_pilot_10k/assembled/assembly.json. Index build scratch and labels
are unnecessary for frozen inference. Do not publish this transfer bundle.

The parallel runner reuses predict.run unchanged and splits S1 into disjoint
contiguous ranges. On an 8-vCPU host start with four processes (the existing
scorer uses two LightGBM threads). The pilot must use a new directory so cached
work cannot inflate throughput. Its two TSV hashes must equal the frozen laptop
pilot, and the full run requires a matching host, worker count, runner, model,
index and unchanged inference code. The existing quality/time/disk/RAM gate
still applies; memory detection now uses the already installed psutil on Linux
and Windows. Stage timing sums are aggregate worker time; the projection uses
measured parallel wall time.

From the transferred project root on Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/python -m unittest discover -s code/business_entity_resolution/src -p 'test_*.py'
.venv/bin/python -u code/business_entity_resolution/src/predict_parallel.py --workers 4 --output-dir artifacts/cloud_pilot
.venv/bin/python -u code/business_entity_resolution/src/predict_parallel.py --workers 4 --limit 1732544 --pilot-dir artifacts/cloud_pilot --output-dir artifacts/cloud_full
.venv/bin/python code/business_entity_resolution/src/assemble_submission.py --run-dir artifacts/cloud_full
.venv/bin/python -X utf8 code/business_entity_resolution/src/validate_outputs.py --output-dir output
```

Stop if any command fails. No full output is submission-ready until official
validation passes. No measured cloud runtime exists yet. AWS Virginia quota was
8 standard on-demand vCPUs at the live check; the proposed 16-vCPU instance does
not fit that quota. No instance was launched. Browser connectivity then failed.
The user approved a US$30 total AWS ceiling; verify the actual instance price,
configure a bounded shutdown, and preserve results before cleanup. This budget
is not an AWS-enforced spending cap.
