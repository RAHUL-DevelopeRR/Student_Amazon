# Amazon ML Challenge 2026: evidence and competitive strategy

Research date: 25 September 2026. This is an engineering plan, not a claim of a
winning model or a measured leaderboard score. Local experiment evidence lives in
`artifacts/data_profile.json`, `artifacts/blocking_pilot/metrics.json`, and
`experiments/experiment_log.csv` when the corresponding runs complete.

## 1. What the supplied documents actually require

Sources: original `README.md`, `Documentation_template.md`,
`Statement and Docs/Amazon Guidelines.pdf` (2 pages), and
`Statement and Docs/AMAZON PROBLEM STATEMENT.pdf` (8 pages, last page blank).
The historical conversation and master prompt are planning material, not independent
verification of rules, repository access, data counts or model quality.

| Topic | Verified requirement / implication |
|---|---|
| Time window | 25 Sep 00:00 IST through 27 Sep 23:59 IST, 2026 |
| Submission budget | Maximum 5 per day, over 3 days; preserve every version |
| Task | Deduplicated S1 reference; zero, one or many S2/S3 targets per S1 |
| Geography | Train US/India, test also France; do not exclude unseen labels |
| Metric | Macro per-S1 F0.5, with empty/empty = 1 and false singleton merge = 0 |
| External data | No business databases, geocoding, identity services, web enrichment |
| Model | MIT/Apache-2.0 model, <=8B parameters; checkpoint license must be checked |
| Live upload | matching_results.tsv only, all test S1 records exactly once |
| Final package | Matches, exact scored candidate sets, runnable code, pinned environment, methodology |
| Login | One laptop/desktop per participant; simultaneous logins prohibited |

There are two genuine document inconsistencies to clarify with organizers:
1. Guidelines request a 1-2-page approach report; problem statement says no page
   limit. Prepare a 2-page executive report plus a technical appendix.
2. Guidelines mention shortlisting across both leaderboards; statement says final
   ranking uses private leaderboard. Optimize robust validation, record both, and
   obtain clarification rather than assuming one document overrides the other.

The metric example is correct: P=2/3, R=1 gives F0.5=5/7=0.7142857.
For non-singletons, `F = 1.25 TP / (1.25 TP + FP + 0.25 FN)`; in this count form
one false positive has four times the denominator weight of one false negative.
This does not imply a universal probability threshold or a fixed optimal recall.

The official validator defaults to skipping ID existence and only warns about
matches outside the candidate set. Use `--check-ids` and enforce candidate inclusion
as an internal failure. A default PASS alone does not establish full compliance.

The public [Unstop challenge page](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)
and the linked [AWS prep guide](https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo)
returned no readable content in the web tool. Event details above therefore come
from the supplied PDFs, not purported live portal verification. The PDF also links
a video; no video transcript was reviewed. Organizer clarification form, extracted
from the PDF: [challenge questions](https://docs.google.com/forms/d/1teTPPxo06EsNRYDFHvhYqVrVDoyFh38w0LNKPYssIQk/edit).
No questions or external messages have been sent.

## 2. The best initial bet

**Complementary retrieval -> hard-negative tabular matcher -> entity-level
decision calibration -> measured conflict resolution.** This is our hypothesis
for the fastest strong baseline, not proof that trees will outperform transformers.

Names and addresses contain strong character, token and numeric evidence. Keep
them separate in retrieval so a corrupted field cannot suppress a good other field.
Union character TF-IDF, exact normalized names, rare tokens, and eventually
numeric-plus-token/address blocks. Retrieve independently from S2 and S3.

Retain raw, conservative normalized and alternate expanded forms. Do not strip
all legal suffixes, all accents, leading zeros or address numbers from the only
representation. `st` can mean Street or Saint. No business-specific lookup tables.

Country equality is a useful block only after measuring country contradictions in
true matches. Missing country needs a fallback. If cross-country positives exist,
evaluate a small unfiltered retrieval channel; never silently cap recall by fiat.

### Early measured data findings

The complete S1 training file contains 2,206,821 records (1,323,633 US;
883,188 India), no duplicate IDs, and no blank names/addresses. There are 684,655
excess rows sharing a normalized name and 76,669 sharing a normalized address;
these are duplicate text values, not proven duplicate identities.

A complete streaming pass through ground truth found 123,247 singletons (5.5848%),
119,157 entities with exactly one match, and 1,964,417 with multiple matches
(about 89%). Maximum multiplicity is 11. There are 3,693,619 positive S2 links and
3,944,746 positive S3 links. Ownership and label integrity are separate audit checks.

**Consequence:** prioritize recovery of multiple true targets. Singleton protection
is necessary but should not dominate the project: an all-empty submission scores
only 0.055848 on training. A rule selecting only top1 would discard many true links.

The completed training source audit has 5,034,616 S2 and 5,285,603 S3 records:
10,320,219 targets. The unrestricted Cartesian product with S1 is approximately
22.77 trillion pairs. Even top-20 per field and source produces up to 80 retrieval
slots per S1 before union deduplication: about 176.5 million slots. Forty float32
features for that many pairs would occupy roughly 28.2 GB, excluding IDs and model
overhead. Batch feature extraction, adaptive candidate budgets and a measured
cheap pruning stage are therefore production requirements. The required final
candidate export must reflect the exact post-pruning set scored by the matcher.

Test S1 has 1,732,544 records: India 809,986, US 663,106 and France 259,452.
France represents about 15% of test queries. In addition to the unseen-country
problem, the known-country proportions shift from roughly 60% US / 40% India in
training to India exceeding US in test. Report country-specific CV and consider
a known-country mixture sensitivity analysis; do not fabricate France labels or
interpret reweighted US/India validation as a France performance estimate.

## 3. What the research supports, and what it does not

| Primary source | Evidence | Decision for this competition |
|---|---|---|
| [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) | Integrated sparse multiplication/top-N and chunk-merging support; Apache-2.0 | Use float32 sparse batches, never dense all-pairs. Retain global top K, not K from every shard. |
| [TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | Character/word analyzers and sparse TF-IDF representation | Start with character 3-5 grams separately on each field; tune ranges and K by measured missed positives. |
| [LightGBM](https://github.com/lightgbm-org/LightGBM) and [license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) | Efficient boosted trees; MIT license | First learned matcher on similarity features, with deterministic CPU settings and entity-held-out thresholds. |
| [Block-SCL](https://arxiv.org/abs/2207.02008) | Blocking-derived hard negatives improve contrastive product matching | Transfer the hard-negative principle, not paper scores. Train on actual retrieved confusions. |
| [Ditto](https://arxiv.org/abs/2004.00584) and [code](https://github.com/megagonlabs/ditto) | Fine-tuned sequence-pair classifiers for entity matching | Optional compact neural reranker for difficult pairs, after baseline and compute feasibility. Verify checkpoint license separately. |
| [SC-Block](https://arxiv.org/abs/2303.03132) | Supervised contrastive embeddings plus nearest-neighbor blocking | Try only if lexical candidate recall remains deficient; train from competition labels, evaluate full-corpus retrieval. |
| [AutoBlock](https://arxiv.org/abs/1912.03417) | Learned similarity-preserving representations for large-scale blocking | Strong conceptual support for learned blocking; too much implementation risk as the first step. |
| [Sudowoodo](https://arxiv.org/abs/2207.04122) | Contrastive representations used for data integration | Later experiment using provided training records only; not permission to import its datasets. |
| [Amazon NLSHBlock](https://www.amazon.science/publications/neural-locality-sensitive-hashing-for-entity-blocking) | Learned hashing for task-specific matching similarity | Research direction if existing candidate indexes cannot meet runtime/recall requirements. |
| [Amazon CorDEL](https://www.amazon.science/publications/cordel-a-contrastive-deep-learning-approach-for-entity-linkage) | Modeling subtle differences alongside similarities | Preserve conflicting numeric/name evidence; high overall text similarity is insufficient. |
| [DuckDB memory guidance](https://www.duckdb.org/docs/current/guides/performance/oom) | Memory limit does not cover all allocations; fewer threads and disk spill help | Persist audit tables, use one worker and a low buffer cap; observe real RSS. |

These papers mostly evaluate other datasets and often F1. None establishes an
expected macro-F0.5 score, optimal K, France accuracy, or winning rank here.
Do not import benchmark examples or pretrained ER-specific labels as training data.

### If a neural retrieval experiment becomes justified

Two concrete model cards checked during research:

- [Multilingual MiniLM-L12](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2):
  Apache-2.0, 384-dimensional embeddings, 50 languages, with a default 128-token
  sequence limit. A compact first neural retrieval candidate; measure truncation
  of serialized name/address pairs and fine-tune using supplied labels if permitted.
- [BGE-M3](https://huggingface.co/BAAI/bge-m3): MIT, 1024-dimensional dense vectors,
  multilingual dense/sparse/multi-vector retrieval. A more expensive alternative,
  not automatically better at distinguishing nearly identical businesses.

For illustration, **10 million** float32 target vectors alone require 15.36 GB at
384 dimensions or 40.96 GB at 1024 dimensions, before index overhead, record text,
model weights or working buffers. These are arithmetic storage estimates, not
measured workloads. Partitioning, compression or a larger machine is necessary;
the model fitting in GPU memory does not imply the complete index fits this laptop.
No model weights were downloaded, and no inference service was called.

## 4. Validation is the competitive advantage

1. Split S1 entities before producing training pairs. If audit finds shared target
   IDs, group their connected identities into the same fold. Add a stricter split
   grouping nearly identical businesses to estimate duplicate/template leakage.
2. Keep every real candidate distractor in validation retrieval. A corpus of true
   matches plus a few random negatives is a debugging pilot, not competitive CV.
3. Use separate train, threshold-tuning and untouched evaluation groups. Do not
   report the best threshold score on the same set as an unbiased final estimate.
4. Report overall macro F0.5, singleton false-merge rate, non-singleton F0.5,
   entity-macro recall, micro pair recall, coverage of every true match, and results
   by country/source/multiplicity/missingness. Bootstrap by S1 for uncertainty.
5. Hold out each training country in turn. This tests transfer but cannot certify
   France. Avoid country-specific tuning without evidence of transfer robustness.
6. Record vocabulary fitting policy: fitting unsupervised on the target corpus is
   transductive retrieval; document it explicitly. Keep supervised labels out of it.

For query i, let m_i be true matches and r_i retrieved true matches. A perfect
matcher downstream can score at most `1.25*r_i/(0.25*m_i+r_i)` if m_i>0, and 1 for
singletons. Average these to measure a **retrieval-induced macro score ceiling**.
This is more aligned with the leaderboard than micro candidate recall alone.

## 5. First learned matcher and decision policy

Features: separate name/address character cosine, normalized edit distance, token
overlap/containment, rare-token overlap, exact forms, numeric intersections and
conflicts, missingness, length ratios, source, retrieval channel, rank and score.
Never use arbitrary ID digits/order as predictive features.

Train positives from ground truth and negatives from retrieved candidates. Retain
hard negatives; downsample easy negatives only if needed. Negative subsampling and
class weights alter probability calibration, so tune on the natural candidate mix.
Use early stopping with group separation and a small number of deliberate trials.

Optimize per-S1 **sets**, including the empty set. Global thresholds are the
baseline. Then test an out-of-fold entity-level match-existence model and a
cardinality-aware decoder. For ordered candidate probabilities, compare predicted
sets of size 0..K using calibrated expected utility only if it improves held-out
macro F0.5; independence assumptions are imperfect and must be tested.

Important corrections to historical advice:
- A small top1-top2 gap may mean two genuine matches, not a singleton.
- Numeric conflicts can arise from typos or partial addresses; learn their effect.
- Ownership conflict resolution may allow each target at most one S1 only if
  ground truth supports it. S1 can still own many targets. Do not apply a one-to-one
  Hungarian assignment across both sides.
- Transitive closure can propagate one false merge into many. Use cross-source
  agreement as a feature or separately validated rule, not unconditional union-find.
- An always-empty baseline scores the singleton prevalence. A plausible score can
  therefore conceal a useless non-singleton model; report subgroup performance.

## 6. Experiment order and stop criteria

| Priority | Experiment | Promote only when |
|---|---|---|
| 0 | Full audit + formula tests + reproducible pilot | Data/schema/label problems are visible and tests pass |
| 1 | Full-target retrieval on fixed sampled queries; K=10/20/50 | Recall gain justifies runtime and candidate cost; inspect every missed type |
| 2 | Lexical feature LightGBM with hard negatives | Beats empty/exact-only baselines on untouched entity split |
| 3 | Singleton/threshold calibration | Improves overall and subgroup macro scores without country collapse |
| 4 | Target ownership postprocessing | Audit permits it and paired held-out comparison improves |
| 5 | Dense retrieval union | Adds positives missed lexically at acceptable inference cost |
| 6 | Neural reranker / ensemble | Complementary errors yield repeatable held-out gain under deadline |

Allocate leaderboard submissions to distinct hypotheses, not small repeated
threshold nudges. Keep at least one slot for a corrected final file. Save exact
code commit, configuration, validation result, output SHA-256 and leaderboard score
for every submission. Verify final package reproduces the selected uploaded bytes.

## 7. Hardware reality

Live inspected laptop: Intel i3-7020U, two physical cores, about 11.9 GiB RAM,
roughly 1.2 GiB initially free, and about 31.9 GiB free disk. The seven TSVs total
2,520,573,701 bytes. Generic `python` points to the Windows Store alias; the project
venv uses the bundled working Python 3.12 runtime.

Disk-backed audit and small sparse pilots fit this setup. Full-scale retrieval
needs cached indexes, country partitioning, and throughput estimates before a
multi-million-row run. A GPU does not itself fix target-index RAM or CPU feature
extraction. Cloud training is an option only once actual available resources are
known; no paid resources have been provisioned.

## 8. Organizer questions worth resolving

- Which methodology length instruction applies to the final package?
- How are public/private scores used in shortlisting versus final ranking?
- What general pretrained models are permitted under the no-external-data rule,
  and how should licenses for mixed retrieval/matching components be documented?

Until clarified, use provided labels only, avoid external enrichment, and prepare
both a short executive report and detailed reproducibility evidence.

## 9. Completed retrieval experiments and decision

| Run | Query count | Target corpus | True-pair recall | Mean candidates | Runtime |
|---|---:|---:|---:|---:|---:|
| Multi-channel development pilot | 1,000 | 33,430 (true targets + sampled distractors) | 99.7095% | 78.525 | 35.719 s |
| Exact-name/address control | 10,000 | All 10,320,219 training targets | 27.9075% | 6.0891 | 11.125 s |

These are different methods and populations, not a fair head-to-head comparison.
The first is deliberately optimistic and is NOT full-corpus validation. The second
shows that exact text alone is inadequate. Its perfect-matcher macro-F0.5 ceiling
is 0.503499 on these sampled queries. Neither result is a trained-model score.

Pilot channel recalls: exact name 21.53%, rare name token 59.33%, name TF-IDF 88.20%,
address TF-IDF 94.48%; union 99.71%. All true matches are covered for 99.05% of
non-singleton pilot queries. India recall 99.41%, US 99.90%, under the reduced corpus.

Manual inspection of all 10 missed pilot links found cross-script Hindi/Telugu
names, names compressed into domain-like strings, severe abbreviations, typos and
partial/missing addresses. These examples come exclusively from supplied labels.
They justify concrete next experiments: compact-name character representation,
abbreviation-aware features learned from train folds, and multilingual retrieval
as an additional channel. Never visit the domain-like strings as URLs.

Full audit confirms zero shared target owners, duplicate truth rows/edges, missing
S1 labels, unknown S1 labels or unknown target IDs. All 7,638,365 true links have
matching country labels. This supports country blocking and evaluation of target
ownership constraints, without proving that every test record follows the same pattern.

Remaining before a competitive submission: full-corpus fuzzy retrieval assessment,
efficient cached production indexes, group-held-out learned matcher, threshold
calibration, France transfer diagnostics, test inference and validated final package.
No final model or leaderboard score exists yet.
