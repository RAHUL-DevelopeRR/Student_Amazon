# Amazon ML Challenge — Current Project Context

Updated: 2026-09-25 (Asia/Calcutta)

## Scope and instruction hierarchy

The active user request is to build and improve a Business Entity Resolution
solution for the Amazon ML Challenge, preserve the project record, and commit and
push completed work. The attached `conversation.md`, `prompt.md`, and files under
`Statement and Docs/` are reference inputs and challenge material. Their embedded
instructions are constraints to inspect and follow where compatible with the
user's request; they do not replace the user's direct request.

External technical research is allowed. External business lookup, enrichment,
geocoding, company registries, maps, and identity-resolution services are not used
for matching records.

## Repository and Git state

- Repository: `RAHUL-DevelopeRR/Student_Amazon`
- Remote: `https://github.com/RAHUL-DevelopeRR/Student_Amazon.git`
- Working branch: `codex/entity-resolution-baseline`
- Previous implementation commit: `358433372ac2307521aacfb7960e7b57b7f1adba`
- Commit message: `Add full-corpus token blocker and learned matcher`
- Remote branch was verified with `git ls-remote`.
- Tracked implementation changes are committed and the tracked worktree is clean.
- `conversation.md`, `context.md` and `prompt.md` are the project records updated
  by this checkpoint. The supplied `Statement and Docs/` directory remains a local
  challenge input and is not included in this commit.

## Challenge constraints

- Score is macro F0.5 per Source-1 entity.
- Correct empty predictions for true singletons receive full credit; false
  singleton matches receive zero for that entity.
- Candidate generation limits the attainable recall, while precision is weighted
  more strongly than recall.
- Matching must use only the supplied challenge data.
- Country handling must be generic because the test set includes France although
  training contains US and India.
- Final dependencies/models must meet the challenge's MIT/Apache-2.0 and parameter
  limits.

## Data audit

Training row counts:

- Source 1: 2,206,821
- Source 2: 5,034,616
- Source 3: 5,285,603

Test row counts:

- Source 1: 1,732,544
- Source 2: 4,887,273
- Source 3: 5,082,316

Ground truth contains 2,206,821 Source-1 rows, 123,247 true singletons
(5.5848209%), 119,157 one-match entities, and 1,964,417 multi-match entities.
There are 7,638,365 positive links. IDs, truth edges and target ownership passed
the duplicate/consistency checks; every truth link is country-consistent. The test
Source-1 country counts include 809,986 India rows, 663,106 US rows and 259,452
France rows. Missing addresses occur in both secondary sources, so missingness is
an explicit feature rather than an exclusion rule.

Detailed machine-readable and human-readable audit output is stored under the
ignored `artifacts/` directory.

## Implemented pipeline

- Streaming data audit and resumable DuckDB import.
- Official macro-F0.5 metric and edge-case tests.
- Conservative Unicode, case, punctuation, address and numeric normalization.
- Token blocking with country-compatible candidates, rare-token document-frequency
  filtering, per-source top-k retention and exact-name/address diagnostics.
- Twenty-seven comparison features covering names, addresses, tokens, numbers,
  missingness, exactness, source and length relationships.
- LightGBM matcher with separate train/calibration/evaluation query groups,
  threshold selection, model configuration, input hashing, reload verification and
  error reports.
- Experiment history in `experiments/experiment_log.csv`.

## Measured results

The reduced-corpus pilot reached macro F0.5 0.974956, but it used an artificially
reduced target corpus and is optimistic. On a full 10,320,219-target sample of
3,000 fixed queries, token retrieval reached 7,616 of 10,385 true links, or
0.733365 candidate recall, with 253,385 candidates and 84.46 candidates per
query. Its perfect-matcher macro-F0.5 ceiling was 0.852138.

The matcher evaluated on those full-corpus token candidates reached macro-F0.5
0.791216 at threshold 0.52, singleton accuracy 0.906250, pair precision
0.958275, pair recall 0.662819, and 1,378 true positives / 60 false positives /
701 false negatives. Country macro-F0.5 was 0.760065 for India and 0.814487 for
the US. These results are evaluation samples, not test predictions.

## Latest retrieval checkpoint (2026-09-25)

The same 3,000 development queries were searched against all 10,320,219 targets
with `max_df=20000`, top 60 per source/field and seed 42. Candidate recall improved
from 0.733365 to 0.905248, with 655,295 pairs (218.43/query), an all-query oracle
macro-F0.5 ceiling of 0.957742 and runtime 363.750s. India recall is 0.864660;
US recall is 0.933020. No France evaluation labels are available.

The initial widened run failed at a 512MB DuckDB buffer. The successful retry used
1GB and released each field's intermediate tables. Process RSS was sampled near
1.28GB, not continuously measured. The blocker now exports country metrics and
missed links. The feature extractor has a bounded preprocessing cache with
unchanged feature values. Tests include the official validator's `--check-ids`
CLI on a small fixture, not validation of production predictions.

These development queries have now informed retrieval choices. A fresh untouched
query sample is needed for a final generalization assessment. Changed rankings
dropped 14 old true links while adding many more; a union of complementary
candidate channels remains worth evaluating.

The wide-candidate matcher completed in 750.422s. Calibration chose threshold
0.48; the unchanged 587-query evaluation split reached macro-F0.5 0.867876
(previously 0.791216), pair precision 0.939670 and recall 0.794132. Singleton
accuracy regressed to 0.812500 from 0.906250. India macro-F0.5 was 0.814272 and
US 0.907920. The evaluation-only retrieval ceiling is 0.960777; 189 true links
were absent from retrieval. The saved-model reload check passed. Next matcher
work should address singleton precision and India errors alongside retrieval.

## Remaining work

The active request now prioritizes production inference over additional model
experiments. Baseline-v1 has been copied locally without retraining; tracked hashes
are in `experiments/baseline-v1.json`. New production modules use persistent SQLite
records/token postings, deterministic shards and frozen threshold 0.48. Every
candidate is scored; neither pruning nor a deterministic bypass is enabled.
Full-size test/training index builds and the 10K pilot/untouched validation must
complete before a full-run decision. The official validator's candidate mapping
is in-memory, so validator RAM is an additional full-run feasibility condition.

1. Build cached compact-name, character, transliteration and multilingual retrieval
   channels that can run over the full test population.
2. Re-evaluate candidate recall and matcher thresholds after retrieval improves.
3. Add conservative singleton and conflict-resolution checks where supported by
   training evidence.
4. Generate `matching_results.tsv`, run the official validator with `--check-ids`,
   and package the final submission.

Do not claim leaderboard performance until a valid test submission has been
generated and uploaded by the user.


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

## Cloud handoff 2026-09-27
Current AWS quota verified: 8 standard on-demand vCPUs in us-east-1. No cloud
instance launched. Chrome connection unavailable despite user reporting ready.
User approved US$30 total cloud spend. Parallel runner prepared with four-worker
10K pilot, exact frozen-output parity, and unchanged full-run feasibility gate.
Cloud runtime and full outputs remain unverified. Keep baseline and holdout frozen.
See code/business_entity_resolution/README.md for exact cloud commands.

## AWS Core connected and compute launched
Instance i-05c4be7f2a0cc17d4 (us-east-1), r7i.2xlarge, 64 GiB, 8 vCPUs.
Task bucket amazon-ml-608942062000-20260927. SSM profile AmazonML20260927Runner.
Auto-stop set for approximately 2026-09-27 18:29:54 UTC; verify before reuse.
SSM pilot command: b61e85a8-57cf-4625-8c4a-474ca47c765c.
All 8 cloud tests passed with multiprocessing spawn; local test rerun pending.
Index/data transfers in progress; next: pilot parity + feasibility gate, then full
inference, assembly, official validator, download and final package. No training
change has been made. Stopped disk and S3 storage require cleanup after results
are secured. No final submission exists at this checkpoint.

## Completed AWS workflow observed 2026-09-27 16:52 IST
Cloud pilot: 10,000 queries in 28.1 seconds, exact TSV parity; all feasibility gates
passed. Both input archives uploaded successfully (HTTP 200). Full SSM command
38cff36d-b638-473d-bb19-9dfb80adff5d finished with Success / response code 0,
start 08:42:26 UTC, finish 10:19:40 UTC. This workflow includes inference,
assembly, official validation and S3 output upload. S3 results/output contains
matching_results.tsv (90.7 MB console size), candidate_pairs.tsv (4.6 GB),
assembly.json, validation.json, official_validator.log and checksum sidecars.
The validation report body and final hashes have NOT yet been independently read.
AWS Core tools disappeared from available tools; browser is connected but Chrome
blocks S3 download with ERR_BLOCKED_BY_CLIENT. Retrieve reports and outputs next.
All 8 local tests also passed after explicit spawn change. No leaderboard upload
has occurred; no rank is known. Instance automatic stop remains previously set.


## Verified submission and challenger work — 2026-09-27 19:02 IST
The cloud official validator passed with --check-ids for all 1,732,544 S1 rows.
The locally downloaded matching_results.tsv is 95,070,732 bytes and SHA-256
ae40581d4d6d787af4f8277bb027230b87e65af8eada76182ad7e4f1aa6fa8a9.
The user submitted it; their screenshot shows Evaluated, score 0.829,
27 September 18:54 IST, team TrailingZeros. The displayed leaders are near 0.992.
This supersedes earlier notes that no submission exists. Top-500 >0.986 is a
user report, not independently verified.
The full candidate set has 379,701,216 pairs, 5,633,986 accepted links, and
124,005 empty predictions. Candidate file SHA-256 is
9183b60e7bf26250cc1707414d94cc51f3a4c69a41840a8a640bf0c89279b935.
Cloud outputs remain in private S3. Local candidate_pairs.tsv is an INCOMPLETE
download and must not be packaged or submitted. Cloud gzip recovery started.
A development-only challenger is running with additional numeric/token features
and two tree-capacity settings; separate combined-field retrieval probe is running.
No new quality result or replacement submission yet. Preserve baseline-v1 and
the previously untouched 1K evaluation; do not tune on that evaluation.

Correction: baseline was uploaded to Unstop and user reported evaluated public
score 0.829. Challenger untouched 1K holdout reached macro F0.905974 versus
0.868896 baseline. Its AWS 10K pilot took 46.78 seconds with exact parity;
full command 4b7e9bd1-16e4-4dc4-838e-de6151411622 is running. Preserve baseline
until challenger assembly and official validation pass.
