# Antigravity handoff: Amazon ML Challenge

## Working directory

Repository root: `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource`

Model and pipeline source: `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource\code\business_entity_resolution`

Frozen submitted baseline model (ignored local artifact): `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource\artifacts\baseline-v1`

Frozen challenger model (ignored local artifact): `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource\artifacts\baseline-v2`

Submitted output: `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource\output\matching_results.tsv`

Local candidate output is incomplete. The complete cloud candidate file is in
private S3 and was also compressed to `output/candidate_pairs.tsv.gz`; do not
use the short/incomplete `output/candidate_pairs.tsv` as a final package.

## Repository and current branch

- Git remote: `https://github.com/RAHUL-DevelopeRR/Student_Amazon.git`
- Branch: `codex/entity-resolution-baseline`
- Latest pushed baseline commit before this handoff: `3616e117d8b9709a795306af7c99c787effd4830`
- The current uncommitted work includes documentation, challenger experiments,
  and this handoff file. Generated artifacts remain ignored.

## What was done

The full baseline run used AWS account `608942062000`, region `us-east-1`, and
an EC2 `r7i.2xlarge` (8 vCPU, 64 GiB). It used only supplied competition data;
no external business lookup, geocoding, or enrichment was used.

The cloud full run completed inference, assembly, and the unchanged official
validator with `--check-ids` for all 1,732,544 test Source 1 rows. The output
contains 5,633,986 accepted links and 124,005 empty predictions. The official
validator reported PASS.

The Unstop public portal evaluated the submitted `matching_results.tsv` at
**0.829** (screenshot supplied by the user). The screenshot also showed leading
teams near 0.992. This score is authoritative for the submission; local
validation scores are not leaderboard scores.

Submitted matching file SHA-256:
`ae40581d4d6d787af4f8277bb027230b87e65af8eada76182ad7e4f1aa6fa8a9`

Complete cloud candidate file SHA-256:
`9183b60e7bf26250cc1707414d94cc51f3a4c69a41840a8a640bf0c89279b935`

Baseline untouched 1,000-query validation against the full 10,320,219-record
training target corpus: macro F0.5 `0.872322`, pair precision `0.939697`, pair
recall `0.793886`, singleton accuracy `0.826923`. This is validation evidence,
not a leaderboard estimate.

## Challenger result

The challenger adds text/number evidence and a joint name/address token
retrieval probe. It was selected using development calibration only. On a new,
untouched 1,000-query training holdout (queries 4001-5000), it achieved:

- baseline macro F0.5: `0.868896`
- challenger macro F0.5: `0.905974`
- challenger pair precision: `0.952117`
- challenger pair recall: `0.849462`
- challenger candidate recall: `0.926184`

This is a measured improvement on held-out supplied data, not evidence that
the public score will reach 0.986. A fresh AWS pilot is currently being run;
use its measured wall time and the feasibility gate before starting a full
replacement inference.

Corrected AWS pilot: 10,000 queries, 2,283,028 candidate pairs, 46.78 seconds
wall time, and exact serial/parallel parity. Linear full projection: 8,105
seconds. First full challenger command 4b7e9bd1-16e4-4dc4-838e-de6151411622
failed only because of shell quoting. Corrected command
bf6e5c7c-f728-4b17-8308-9c8dec814507 is InProgress under
/opt/amazon-ml-challenger; baseline remains preserved.

## Important files

- `code/business_entity_resolution/src/token_blocking.py`: training retrieval.
- `code/business_entity_resolution/src/train.py`: baseline LightGBM matcher.
- `code/business_entity_resolution/src/build_test_candidates.py`: disk-backed
  test index and token retrieval.
- `code/business_entity_resolution/src/predict.py`: bounded frozen inference.
- `code/business_entity_resolution/src/predict_parallel.py`: spawn-based AWS
  parallel runner.
- `code/business_entity_resolution/src/assemble_submission.py`: exact output
  assembly and candidate containment checks.
- `code/business_entity_resolution/src/validate_outputs.py`: output checks plus
  unchanged official validator.
- `code/business_entity_resolution/src/challenger_features.py`,
  `train_challenger.py`, `validate_challenger.py`, `probe_joint_retrieval.py`,
  `run_challenger.py`: challenger experiment and gated cloud runner.
- `Documentation_template.md`: filled methodology for team TrailingZeros.
- `PROGRESS.md`, `context.md`, `conversation.md`, `prompt.md`: accumulated
  progress and conversation context.

## Team

Team: `TrailingZeros`
Members: `Rahul S`, `Kumaravel K`, `Sibidharan S`, `Rishanth P`

## Reproduction

Run tests:

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s code/business_entity_resolution/src -p 'test_*.py'
```

The README in `code/business_entity_resolution/` contains the baseline
reproduction commands. Do not rerun a full local inference on this laptop.
Keep source TSVs and `utils/validate_submission.py` unchanged. Do not commit
`artifacts/`, `output/`, caches, model binaries, or temporary transfer files.

## Decision record

The first submission is preserved. The challenger may replace it only if the
AWS pilot completes, passes serial/parallel parity, and its measured full-run
projection fits the remaining competition time with assembly and validation
headroom. Never call a reduced pilot a full-corpus score or a leaderboard rank.
# Latest verified state — 27 September 2026, 22:19 IST

Open `C:\Users\DELL\Downloads\6ab10eb3b23ba_student_resource` in the IDE.
The existing public score is still 0.829 (verified live in Unstop).

Baseline final package is complete at `output/TrailingZeros_baseline_submission.zip`:
2,215,069,858 bytes, SHA-256
`0f66d5b07cdd56a76c34f81b4896eef6146407a69d00cdde36d68e3f8d96c764`.
Both TSV hashes match cloud validation, ZIP CRC passed, and selected weights,
source, reproduction command, methodology and official validator are included.
The local baseline candidate TSV is complete; older incomplete-download warnings
below are superseded. The ZIP has not been submitted.

Challenger inference is active on `i-05c4be7f2a0cc17d4` in `us-east-1`, root
`/opt/amazon-ml-challenger`. Resume command:
`da8f4925-2ca0-4b88-bc7a-a170434cb634`. This corrects a short SSM timeout and
preserves completed shards. Do not launch a concurrent run or delete shards.
Validator source/input symlinks have been restored. Cloud packaging command
`1ffcee9e-a149-45a6-8fc6-217b9c1c899c` waits for full validation and saves
`results/challenger-final/TrailingZeros_submission.zip` in the existing private
bucket `amazon-ml-608942062000-20260927`. TSVs/reports go to
`results/challenger-output/`. Download and verify before using them.

Instance shutdown is scheduled for approximately 23:59:54 IST. Unstop's matching
upload closes at 23:59 IST. A separate ZIP submission route is not visible; the
user does not know one. Never claim an upload without its portal receipt.
No new training/scoring changes were made in this continuation. Eight regression
tests passed; the fresh challenger validation remains macro F0.5 0.905974,
precision 0.952117, recall 0.849462 on 1,000 queries against 10,320,219 targets.
Neither 0.999 nor a top-percentile finish is established.
