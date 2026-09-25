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

## Remaining work

1. Build cached compact-name, character, transliteration and multilingual retrieval
   channels that can run over the full test population.
2. Re-evaluate candidate recall and matcher thresholds after retrieval improves.
3. Add conservative singleton and conflict-resolution checks where supported by
   training evidence.
4. Generate `matching_results.tsv`, run the official validator with `--check-ids`,
   and package the final submission.

Do not claim leaderboard performance until a valid test submission has been
generated and uploaded by the user.
