# Progress: 25 September 2026

## Completed
- Read attached planning files, README, methodology template, both official PDFs,
  and validator; rendered and inspected PDF pages.
- Researched primary papers, library documentation, and model cards. See
  RESEARCH_AND_STRATEGY.md for sources, decisions and rule ambiguities.
- Created Python 3.12 venv and pinned installed dependencies; pip check passes.
- Audited all seven TSVs with SHA-256 fingerprints and disk-backed aggregates.
- Implemented conservative normalization, macro-F0.5 evaluator and five runnable
  tests including a complete miniature audit, metric grid and retrieval checks.
- Implemented chunked name/address character TF-IDF retrieval, exact-name and
  rare-token union, country compatibility and per-source global top-K.
- Ran combined retrieval pilot and full-target exact-match diagnostic.

## Measured findings
- Train: 2,206,821 S1; 5,034,616 S2; 5,285,603 S3.
- Test: 1,732,544 S1; 4,887,273 S2; 5,082,316 S3.
- 5.5848% training singletons; 89.0157% have multiple matches; maximum 11.
- 7,638,365 positive links; no ownership, ID, duplicate-label or country violations.
- France: 259,452 test S1 entities, 14.98% of test.
- Pilot combined recall 99.7095% on 1,000 queries / 33,430 targets; optimistic,
  not a full-corpus or held-out matcher score.
- Full-target exact-name/address recall 27.9075% on 10,000 queries / 10,320,219 targets.

## Outputs
- artifacts/data_profile.json and artifacts/data_profile.md
- artifacts/blocking_pilot/{metrics.json,candidate_pairs.tsv,missed_pairs.json,missed_examples.json}
- artifacts/blocking_exact_full/{metrics.json,candidate_pairs.tsv,missed_pairs.json}
- experiments/experiment_log.csv

## Validation and runtime
- Five unittest tests pass; pip check passes; original data/docs/validator unmodified.
- Scalar DuckDB Python normalization was too slow; replaced with streaming Python
  normalization and bulk import. Audit is resumable per completed source file.
- Audit database is about 2.60 GB; configured DuckDB buffer limit 512 MB, one thread.
  Observed working sets during sampled checks were roughly 100-550 MB; this is not
  a guaranteed or continuously measured peak.
- A simultaneous analysis database open was rejected by DuckDB's file lock. It was
  rerun successfully after the retrieval process exited. Use one writer at a time.
- Generic python command is a Windows Store alias; use .venv/Scripts/python.exe.

## Next required work
1. Measure fuzzy retrieval against full target corpus; cache indexes for throughput.
2. Train LightGBM on hard negatives using identity-group splits.
3. Tune set decisions/empty predictions on separate calibration data.
4. Add compact-name and multilingual channels based on observed missed links.
5. Generate test predictions, run official validator with --check-ids, package.

No neural weights, model training, test predictions, leaderboard upload or Git push
has been performed. Repository visibility was not changed. Existing untracked user
documents remain untouched.
