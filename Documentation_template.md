# ML Challenge 2026: Business Entity Resolution

**Team Name:** TrailingZeros
**Team Members:** Rahul S, Kumaravel K, Sibidharan S, Rishanth P
**Submission Date:** 27 September 2026

## 1. Executive Summary
Our submitted baseline uses disk-backed lexical candidate retrieval followed by a
supervised LightGBM pair classifier. It preserves multiple matches and empty
predictions and processes every test Source 1 entity, including France.
This document describes the submitted baseline; experimental challengers are not
part of its predictions.

## 2. Methodology
### 2.1 Problem Analysis
Names vary in spelling, word order, spacing, suffixes and script. Addresses have
missing fields, reordered tokens, number changes and abbreviations. Similar names
and addresses can describe different businesses, so false merges matter.
### 2.2 Solution Strategy
We normalize Unicode with NFKC and casefold, replace punctuation with spaces,
and retain separate accent-folded and limited abbreviation-expanded comparisons.
The supplied source data and training labels are the only business data used.
No business lookup, external enrichment or geocoding was used. Country remains
an open string label; a missing country is compatible with any label.

## 3. Candidate Generation
SQLite stores records and inverted postings with uint32 target ordinals. Each
field uses distinct tokens of at least three characters whose target document
frequency is at most 20,000. Shared tokens contribute log(1 + N/df); the top 60
are retained independently for name/address and S2/S3, after country filtering.
Exact normalized names and addresses add targets when the entire exact block
has at most 100 members. Oversized exact blocks are skipped, not truncated.
The final union is deduplicated and every member is scored by the classifier.
The exported candidate_pairs.tsv contains exactly these scored candidates.

The full test run contains 1,732,544 queries and 9,969,589 target records. It
scores 379,701,216 candidate pairs (219.16 per query approximately). Retrieval
is bounded by postings and batches, rather than a dense all-pairs matrix.
Recall is measured on training labels, not assumed perfect: the held-out 1,000
training queries achieved candidate recall 0.913537 against all 10,320,219
training targets. Missed candidates remain false negatives in matching metrics.

## 4. Matching Model
The baseline has 27 features: for each name/address field, character ratio,
token-sort and token-set ratios, expanded-text ratio, expanded text without
spaces ratio, accent-folded ratio, token Jaccard, numeric-token Jaccard,
disjoint-number indicator, nonempty exact match, missingness for each record,
and length ratio; the last feature indicates target source S3. Entity IDs and
country identity are not model features.

The self-trained LightGBM binary classifier uses 180 trees, 15 leaves, depth 6,
minimum 40 samples per leaf, learning rate 0.05 and seed 42. LightGBM uses the
MIT license. No pretrained checkpoint or external model weights are used;
the tree model is far below eight billion learned parameters.

3,000 development queries were retrieved against the complete training target
corpus. SHA256 of seed plus normalized name/address/country assigns identity
groups to 60% training, 20% calibration and 20% development evaluation. There
were 1,809 fitting queries and 395,812 fitting pairs (5,601 positives). All
retrieved negative pairs were retained. The independent threshold 0.48 was
selected by calibration macro F0.5. No one-to-one assignment is imposed.

## 5. Results and Error Analysis
An additional 1,000 training queries, outside the development selection and
checked for normalized-identity overlap, were evaluated once with the frozen
policy against the full target corpus. Their macro F0.5 was 0.872322, pair
precision 0.939697, pair recall 0.793886, and singleton accuracy 0.826923 across
52 singletons. India macro F0.5 was approximately 0.83556 and US 0.89755.
France has no labeled validation data; we do not claim a France validation score.
Macro F0.5 is computed separately per S1, including correctly empty predictions
as 1 and false singleton merges as 0.

The public portal evaluated the baseline at **0.829**, submitted 18:54 IST on
27 September (user-supplied portal screenshot). This is distinct from local
validation, and is not a private leaderboard estimate.

Observed failure patterns include false merges between similar businesses at
different street numbers, missing addresses, altered/concatenated names,
transliteration and difficult retrieval. These are limitations, not proven
causes of the public/private distribution gap.

## 6. Runtime and Output Verification
The AWS workflow ran on an r7i.2xlarge CPU instance (8 vCPU, 64 GiB RAM), with
four worker processes and two classifier threads per process. A fresh 10,000
query pilot took 28.1 seconds and matched the laptop pilot TSV hashes exactly.
The full workflow from inference through assembly, validation and S3 transfer
ran 08:42:26–10:19:40 UTC (97 minutes 14 seconds). This is workflow time,
not an isolated training or inference benchmark. Assembly took 759.94 seconds.

The output has 5,633,986 accepted links and 124,005 empty predictions. Country
coverage is India 809,986, US 663,106 and France 259,452. All 1,732,544 S1 rows
passed the unchanged official validator with --check-ids. The additional
assembler verifies coverage, IDs, duplicates and match containment in candidates.
The metric/retrieval/inference regression suite passed eight tests, including
serial/parallel output parity.

Matching SHA-256: `ae40581d4d6d787af4f8277bb027230b87e65af8eada76182ad7e4f1aa6fa8a9`
Candidate SHA-256: `9183b60e7bf26250cc1707414d94cc51f3a4c69a41840a8a640bf0c89279b935`

## Appendix: Reproduction
Source is under code/business_entity_resolution/src with pinned requirements
and commands in its README. token_blocking.py and train.py reproduce training;
build_test_candidates.py creates the index, predict.py/predict_parallel.py
produce scored shards, assemble_submission.py creates both TSVs, and
validate_outputs.py invokes the supplied official validator. Preserve source
TSVs and use tab separators. Generated caches and model files stay out of Git.
