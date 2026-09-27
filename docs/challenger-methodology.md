# ML Challenge 2026: Business Entity Resolution

**Team:** TrailingZeros
**Members:** Rahul S, Kumaravel K, Sibidharan S, Rishanth P
**Date:** 27 September 2026

## 1. Executive Summary
This package uses a self-trained LightGBM pair classifier with 51 lexical and
numeric features, and disk-backed token retrieval with combined name/address
ranking. It permits multiple matches and empty predictions. Country labels are
open strings; all France queries are processed. No external business lookup,
geocoding, enrichment, pretrained weights or test labels are used.

## 2. Candidate Generation
SQLite stores normalized records and inverted postings of target ordinals.
Normalization uses NFKC, casefold and punctuation-to-space conversion. Distinct
tokens of at least three characters with target document frequency <=20,000
contribute log(1 + N/df). Retrieval retains the top 60 separately for name and
address and separately for S2 and S3, after country compatibility filtering.
Nonempty exact normalized names/addresses add complete blocks of at most 100
targets; oversized exact blocks are skipped. A third ranking sums the shared
name and address token contributions and retains its top 60 per target source.
The deduplicated union is the exact candidate set scored by the classifier and
exported in candidate_pairs.tsv. No candidate pruning after scoring is hidden
from that file. This is sparse bounded retrieval, not dense all-pairs scoring.

The complete test population is 1,732,544 S1 queries and 9,969,589 targets.
Actual candidate count, accepted matches, empty predictions, country counts,
assembly time and output hashes are recorded in verification/assembly.json.

## 3. Model and Features
The original 27 features cover character and token similarity, expanded and
accent-folded text, token/numeric Jaccard, disjoint numbers, exact agreement,
field missingness, length ratios and target source. Twenty-four additional
features capture token containment and differences, compact-string similarity,
first-number agreement/conflict, all-number agreement and number-sequence
similarity. The implementation is challenger_features.py. Entity IDs and
country identities are not classifier features.

The selected classifier has 300 boosted trees, <=31 leaves per tree,
learning rate 0.05, minimum 30 examples per leaf, deterministic training and
seed 42. The threshold is 0.475. Parameters are supplied in the packaged
model/model_config.json. Both the team's original source and self-trained
weights are MIT licensed; LightGBM itself is independently MIT licensed:
https://github.com/lightgbm-org/LightGBM
No third-party pretrained checkpoint is used. The model is far below 8B
parameters (at most 9,300 leaf values and 9,000 split thresholds).

Model SHA-256: 90d50f4d2d2b2bc3369002e021fcc30b8b5df27dc63528a24afd27ed988a71fa

## 4. Training and Validation
3,000 development S1 queries were retrieved against all 10,320,219 training
targets. Normalized name/address/country identity hashes with seed 42 assign
60/20/20 fitting/calibration/development evaluation groups. All retrieved
negative pairs are retained; pairs are never randomly split. Training uses
655,295 development candidate pairs across the three groups; 1,809 S1 queries
belong to the fitting group. Two tree capacities were compared and the 300-tree
model was selected using calibration macro F0.5 (0.895612). Its development
evaluation score was 0.892702. Combined-field retrieval was tested only on the
development population before the fresh evaluation.

A separate 1,000-query set, ordinals 4,001–5,000 in the deterministic training
selection, was checked for normalized-identity overlap and evaluated once
against the complete target corpus. No tuning followed this result.

| Metric | Baseline on same fresh set | Selected challenger |
| --- | ---: | ---: |
| Per-S1 macro F0.5 | 0.868896 | 0.905974 |
| Pair precision | 0.939268 | 0.952117 |
| Pair recall | 0.791049 | 0.849462 |
| Candidate recall | 0.908747 | 0.926184 |
| Scored candidate pairs | 220,445 | 227,444 |

The joint comparison took 207.594 seconds. Correctly empty predictions score 1;
false singleton merges score 0. Missing retrieved positives count as false
negatives. France has no labeled training examples, so no France quality claim
is made. These measurements are not public/private leaderboard estimates.
The earlier baseline's public score was 0.829; it is not a challenger score.

## 5. Runtime and Verification
Inference runs on an AWS r7i.2xlarge (8 vCPU, 64 GiB RAM), four worker processes,
two LightGBM threads each. A fresh 10,000-query challenger pilot took 46.783
seconds and scored 2,283,028 pairs; its first 250 queries matched serial output
exactly. The linear inference projection was 8,105 seconds, excluding assembly
and validation. Checkpoint reuse after an interruption is not a fresh runtime
benchmark. Each 250-query shard has a checksum and atomic completion marker.

The package builder requires the unchanged official validator to pass with
--check-ids for the complete output. It hashes the actual bytes while packaging
and compares them to the validation report, then verifies ZIP CRCs. Reports and
the validator log are included under verification/. The regression suite has
eight passing tests, including metric, retrieval, France, empty outputs and
checkpoint behavior. Validation verifies format and IDs, not prediction quality.

## 6. Reproduction and Limitations
Use the packaged README, pinned requirements and selected frozen model. Build
the index from supplied source TSVs; reproduce_submission.py then recreates
both TSVs and validates them. Training is implemented in token_blocking.py and
train_challenger.py; the latter requires the training inference index built by
build_test_candidates.py --split train --data-dir dataset/train --index
artifacts/train_inference_index. The packaged model fixes the selected policy.

Recall remains limited by lexical retrieval, transliteration and missing fields.
False merges remain possible for similar businesses and addresses. Full-corpus
validation difficulty is preserved, but a 1,000-query labeled sample cannot
establish a 0.999 score or leaderboard percentile. No such claim is made.
