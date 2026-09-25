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
1. Add cached compact-name, transliteration and multilingual retrieval channels.
2. Re-run retrieval and matcher evaluation after those channels recover missed links.
3. Generate test predictions, run the official validator with `--check-ids`, and package.

No test predictions, leaderboard upload or repository visibility change has been
performed. The supplied challenge documents remain local inputs.

## Follow-up: learned matcher and realistic retrieval

The initial no-training status above describes the first checkpoint. A learned
LightGBM matcher is now implemented in `src/train.py` with 27 comparison features,
identity-separated train/calibration/evaluation splits, saved model configuration,
candidate-input hash, model reload verification and per-country error reports.
Six tests pass, including token retrieval and learned feature/scoring checks.

Reduced-corpus pilot: 610 training queries / 47,918 pairs, 190 calibration queries,
200 evaluation queries. Calibration chose threshold 0.67. Evaluation macro F0.5
0.974956, pair precision 0.986667, pair recall 0.965217. This remains optimistic
because the target corpus was artificially reduced.

Full-corpus character retrieval on 100 queries was stopped after a throughput
measurement of about 375,000 target names per 45 seconds. No recall result was
produced; see artifacts/blocking_full_100/run_status.json. Production needs cached
indexes, not repeated text transforms.

A new SQL token blocker scanned all 10,320,219 targets for 3,000 fixed queries in
215.985 seconds. It ranks shared tokens with DF <=2000 separately for name/address,
retains top 30 per source/field, and unions exact-name/address blocks. Recall:
0.733365 (7,616 / 10,385 links), 253,385 candidates, 84.46 per query. Perfect-matcher
macro-F0.5 ceiling: 0.852138. This is a full-target sampled-query result, not a
full-production run. Retrieval currently imposes a material accuracy ceiling.

The learned matcher was then run on those full-corpus token candidates: 1,809
training queries / 151,079 pairs / 4,495 positives, 604 calibration queries and
587 evaluation queries. Calibration chose threshold 0.52. Evaluation macro-F0.5
was 0.791216, singleton accuracy 0.906250, non-singleton F0.5 0.784584, pair
precision 0.958275 and pair recall 0.662819 (1,378 TP / 60 FP / 701 FN). Country
macro-F0.5 was 0.760065 for India and 0.814487 for the US. The corresponding
retrieval ceiling is 0.852138, so the remaining gap is primarily missed candidate
links rather than the classifier alone. This is still an evaluation sample, not a
test submission.
