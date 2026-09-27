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
macro-F0.5 was 0.760065 for India and 0.814487 for the US. The 0.852138 retrieval
ceiling above covers all 3,000 queries, while matcher evaluation covers 587 queries;
these different populations must not be subtracted to quantify classifier loss.
This is still an evaluation sample, not a test submission.

## Retrieval follow-up (2026-09-25)

Diagnosis on the fixed 3,000-query sample found that 2,765 of the 2,769 missed
links shared at least one name/address token of length >=3. This motivated a
wider frequency cutoff before introducing embeddings or new dependencies.

With seed 42, `max_df=20000`, `top_k=60` per field/source and the same complete
10,320,219-target corpus, retrieval found 9,401 / 10,385 true links (0.905248),
up from 7,616 (0.733365). Candidate count increased from 253,385 to 655,295;
average count is 218.43. All-query oracle macro-F0.5 rose to 0.957742. Runtime
was 363.750 seconds with a 1GB DuckDB buffer. India recall was 0.864660 and US
recall 0.933020. This is a development-query comparison, not a leaderboard estimate.

The first 512MB attempt failed during address postings allocation; it has no
recall result and is logged separately. The blocker now drops completed field
tables, accepts `--memory-limit`, reports country metrics and exports missed IDs.
Sampled process memory during the successful run was approximately 1.28GB; this
is not a measured peak or a total-process memory guarantee.

Widening retrieval added 425,692 pairs and removed 23,782 through changed ranking.
Fourteen true links found by the old blocker were lost, despite the net gain of
1,785 links. A later union experiment should retain complementary candidates.

The feature extractor now caches up to 8,192 preprocessed strings. All 27 values
matched the previous implementation across a 10,000-pair synthetic comparison;
that repeated-text check took 1.859s before and 0.469s after. This is a synthetic
microbenchmark, not a full-training speedup claim. Six tests pass, including the
official validator CLI with `--check-ids` on the small fixture. Final test-set
output validation remains pending until production predictions exist.

Retrained LightGBM on the wide candidates: 1,809 training queries, 395,812 pairs
and 5,601 positives; 604 calibration queries and 587 evaluation queries. Calibration
selected threshold 0.48. Evaluation macro-F0.5 improved from 0.791216 to 0.867876,
pair precision was 0.939670 and pair recall 0.794132 (1,651 TP / 106 FP / 428 FN).
Singleton accuracy declined from 0.906250 to 0.812500. India macro-F0.5 was
0.814272; US was 0.907920. The same-evaluation-query oracle ceiling was 0.960777,
with 189 links absent from retrieval. Total matcher runtime was 750.422s. Saved
model predictions passed the reload check. These are development comparisons on
reused queries, not a fresh generalization estimate.

Next: improve singleton precision and India errors, evaluate a union retaining
complementary old candidates, build reusable full-population retrieval indexes,
and assess the frozen approach on untouched queries before production inference.

## Production inference implementation (2026-09-26)

The user requested phases A-G: freeze baseline-v1, implement resumable inference,
profile a deterministic 10,000-query test pilot, measure an exact-match cascade,
evaluate untouched training queries, and only run the full submission when feasible.
The existing 0.48-threshold LightGBM model is frozen without retraining; its hashes
are recorded in `experiments/baseline-v1.json`. Generated models remain local.

New scripts implement persistent normalized SQLite records, compact uint32 token
posting lists, deterministic query shards, 32-query scoring batches, atomic output
files, SHA-256 manifests, corruption recovery, frozen-model loading, threshold-only
postprocessing, streaming assembly and official-validator execution. Candidate
rules remain top 60 per source/field, max token DF 20,000, and exact block cap 100.
Every retained candidate is scored. No cascade bypass or extra pruning is enabled.

Test inference reads only test sources and the frozen model. Training validation
is a separate explicit command, rejects development-group overlap and retains
the original full training target population. Pilot outputs are explicitly partial.

The index build initially failed when a 1GB DuckDB buffer attempted to window-sort
all target strings. It now sorts narrow IDs by source and joins payloads afterward.
Completed normalized source imports were reused on restart. An extended machine/
tool interruption and low-memory interval affected elapsed build time; no speed
claim is based on that interrupted build. Test subprocesses initially failed to
start during the low-memory interval; reruns passed after the user freed memory.

At this implementation checkpoint, full-size index builds and pilot/validation
measurements are in progress. No final production output or leaderboard submission
has been generated. The full-run gate also accounts for the official validator's
in-memory candidate mapping, whose memory demand can greatly exceed inference RAM.


## Production inference results (2026-09-27)

Both persistent indexes completed after checkpoint recovery. Test: 1,732,544
queries / 9,969,589 targets; train: 2,206,821 queries / 10,320,219 targets.
Baseline-v1 remains unchanged at threshold 0.48, seed 42, max_df 20,000,
top 60 per source/field and exact-block cap 100. No pruning or bypass was enabled.

The exact 10,000-query test pilot scored 2,186,501 pairs (218.6501/query;
P50/P90/P99 = 237/247/295). Measured seconds: candidate generation 90.858,
candidate loading 107.719, features 336.074, model scoring 21.446, scoring-output
writing 6.041, total 562.465. Candidate JSON writing adds 1.180 seconds already
included in generation. Invocation wall time was 571.516 seconds. Sampled maximum
RSS was 281,714,688 bytes; this is not a continuously measured peak.

Pilot assembly produced 10,000 rows in each TSV, 31,973 matches, 747 empty
predictions, and country counts India 4,662 / US 3,877 / France 1,461. It passed
the unchanged official validator with --check-ids against an explicit 10K S1
subset and all original test targets. This is NOT a full submission. Files:
- artifacts/test_pilot_10k/assembled/matching_results.tsv: 541,774 bytes;
  SHA-256 563a005c39730ee11061c664ee9e950944e60f3cf28690800076615050bd500b
- artifacts/test_pilot_10k/assembled/candidate_pairs.tsv: 28,311,514 bytes;
  SHA-256 1e28cccf14162429610062bb3f5713214cb0606d7473703f846cb31d34eca664

Untouched 1,000-query validation against all training targets: macro F0.5
0.872322; candidate recall 0.913537; pair precision 0.939697; pair recall
0.793886; singleton accuracy 0.826923 (52 singleton queries); India F0.5
0.835561; US F0.5 0.897553; 217.363 candidates/query. No tuning followed this
first result. No France labels are available, so France accuracy is unknown.

The development exact-name AND address rule had 127 TP / 0 FP and zero macro
F0.5 change. It remains disabled. Indexed retrieval exactly reproduced 2,951 of
3,000 prior SQL candidate sets; 949 pairs were added and 957 removed. The exact
cause of those differences has not been established; no parity claim is made.

The gate passed quality and disk but rejected runtime and validator RAM:
27.069 hours projected inference, versus 14.927 hours remaining at the check;
75% headroom plus assembly/validation would need approximately 49.4 hours.
Official-validator candidate memory lower bound: 26,138,643,400 bytes (24.34 GiB),
versus 3,422,281,728 available bytes. Actual validator memory would be higher.
The full run was therefore not started. No complete upload file exists and
nothing was uploaded to Unstop. A faster machine with sufficient validator RAM,
or a separately measured implementation speedup, is required before reconsidering
full inference. Keep this holdout frozen; do not tune on its result.

Eight regression tests passed, including metric/retrieval, checkpoint/corruption,
France/empty output, and official-validator fixtures. Python syntax and pip check
passed. Detailed reproducible measurements are tracked in
experiments/production-inference-20260927.json; large artifacts remain ignored.

## 2026-09-27 cloud continuation
AWS Virginia live quota: 8 standard on-demand vCPUs (0 used). The 16-vCPU proposal
cannot launch within this quota. No paid resource was created. Browser control
became unavailable and stayed unavailable after the user's ready reply.
Added predict_parallel.py to reuse unchanged frozen inference in disjoint worker
processes. It requires a fresh 10K pilot, exact TSV-hash parity with the laptop
pilot and all existing feasibility gates before full inference. Linux RAM check
uses existing psutil. Cloud throughput remains unmeasured; no full submission
or new trained model exists. Run instructions are in the code README.
Validation: all 8 regression tests passed, including serial/parallel TSV hash
parity on the fixture and official validator --check-ids. Live laptop gate still
rejects full runtime/RAM and accepts quality/disk. No cloud timing was measured.
