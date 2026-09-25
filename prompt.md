# Codex Master Prompt — Amazon ML Challenge 2026

```text
You are the engineering agent for my Amazon ML Challenge 2026 project.

Repository:
RAHUL-DevelopeRR/Student_Amazon

Challenge:
Business Entity Resolution.

IMPORTANT COMPETITION RULES:
- Use ONLY the provided Amazon training/test data for entity resolution.
- Never query Google, Maps, geocoding APIs, company databases, government registries,
  external entity-resolution APIs, or any external business identity source.
- External technical documentation/libraries are allowed, but external business data is forbidden.
- Final matching model must satisfy the competition's MIT/Apache-2.0 licensing requirement
  and be <= 8B parameters.
- Do not modify or corrupt the original dataset files.
- All .tsv files must be read with sep="\t".
- The final leaderboard metric is macro F0.5 per Source-1 entity.
- Precision is more important than recall.
- Singletons are extremely important.
- Test includes France although training contains only US and India.
  Country handling must therefore be open-set and generic.

FIRST GOAL:
Do NOT jump directly into a complex neural network.
Build a clean, reproducible competition pipeline first.

Create this structure:

code/
  business_entity_resolution/
    src/
      __init__.py
      config.py
      profile_data.py
      normalize.py
      metric.py
      blocking.py
      features.py
      train.py
      predict.py
      postprocess.py
    README.md
    requirements.txt

experiments/
  experiment_log.csv

output/

artifacts/

Also create AGENTS.md in the repository root describing the challenge rules
and the development workflow.

PHASE 1 — DATA AUDIT

Implement code/business_entity_resolution/src/profile_data.py.

It must efficiently inspect the large TSV files without unnecessary copies.

Analyze:

1. row count of every train/test source
2. exact columns and dtypes
3. memory/file sizes
4. missing business_name counts/rates
5. missing business_address counts/rates
6. country distributions
7. duplicate entity IDs
8. exact duplicate normalized names
9. exact duplicate normalized addresses
10. ground-truth row count
11. number and percentage of Source1 singletons
12. number and percentage with exactly 1 match
13. number and percentage with >1 matches
14. distribution of number of true matches per Source1
15. positive matches to Source2 vs Source3
16. check whether an S2/S3 ID ever maps to more than one S1 in ground truth
17. examples of noisy matched pairs from ground truth
18. name/address length statistics
19. token count statistics
20. country breakdown for matches

Save machine-readable results to:
artifacts/data_profile.json

Save a human-readable report to:
artifacts/data_profile.md

Print a concise summary to the terminal.

PHASE 2 — EXACT AMAZON METRIC

Implement metric.py with Amazon's exact macro F0.5 calculation.

Rules:
- Score independently for every Source1 entity.
- True singleton + predicted empty = 1.0.
- True singleton + any prediction = 0.0.
- Non-singleton should use standard precision/recall and beta=0.5.
- Macro-average across every Source1 entity.
- Include tests for edge cases.

PHASE 3 — BASELINE NORMALIZATION

Implement normalize.py while preserving original data.

Create conservative normalization utilities for:
- Unicode normalization
- lowercase/casefold
- whitespace normalization
- punctuation normalization
- & / "and"
- legal suffix normalization only where safely learned/defined from competition context
- address abbreviation handling
- extracted numeric tokens

Do NOT hard-code logic that only works for India or US.

PHASE 4 — FIRST CANDIDATE GENERATOR

Implement a scalable baseline candidate generator using complementary retrieval:
- country-compatible candidate restriction
- character n-gram TF-IDF for business_name
- character n-gram TF-IDF for business_address
- exact normalized-name blocks where useful
- rare-token matching

Generate top-K candidates independently from name and address retrieval and union them.

Evaluate blocking on TRAIN using the ground truth:
- candidate recall
- percentage Source1 entities with all true matches covered
- average candidates per Source1
- P50/P90/P99 candidates per Source1
- reduction ratio

Candidate recall is the primary Phase-4 metric.

Do NOT generate test predictions yet until the train blocking evaluation is complete.

PERFORMANCE:
The data files are hundreds of MB each, so use efficient code.
Prefer Polars/PyArrow or memory-conscious pandas and sparse scipy/scikit-learn operations.
Avoid constructing all S1 x S2/S3 pairs.
Never create a dense pairwise similarity matrix for the full dataset.

REPRODUCIBILITY:
- random_seed = 42 where randomness exists.
- Log every experiment.
- Add comments explaining non-obvious decisions.
- Keep scripts callable from the repository root.
- Do not alter official utils/validate_submission.py.
- Do not commit generated models, huge caches, or outputs unless explicitly requested.

Before making changes:
1. inspect README.md
2. inspect Documentation_template.md
3. inspect utils/validate_submission.py
4. verify actual Git LFS files are present
5. report the dataset sizes

Then implement PHASES 1-4.

Run the profiling script and metric tests.

At the end report:
- files created/modified
- commands executed
- important dataset findings
- blocking metrics
- any memory/runtime issues
- git status

Commit the working changes with:
"Build entity resolution profiling and baseline pipeline"

Do not invent results. If something cannot be run because of environment/resources,
say exactly what failed.
```

---

# Follow-up prompts captured during execution

## User directives

1. `Continue from where it left`
2. `Continue what is the plan....... DO it`
3. `Continue from where it interrupted and commit and push, after finishing`
4. `Add the entire conversation to conversation.md and context to context.md and prompts to prompt .md`

## Execution prompt for the current checkpoint

Continue the Amazon ML Challenge entity-resolution project from the current
repository state. Preserve the official data, challenge documentation and
validator. Read the existing conversation and prompt files before changing them.
Record the user-visible conversation in `conversation.md`, record the current
technical state, measured results, constraints, Git state and remaining work in
`context.md`, and append the direct user directives plus the active execution
instructions to `prompt.md`. Do not include private chain-of-thought. Preserve
existing content, use plain Markdown, verify all three files after editing, and
report the exact files changed.

## Publication prompt used for the previous checkpoint

Finish the measured implementation, run the compact tests and dependency checks,
commit the implementation and progress documentation on the active competition
branch, push it to `origin`, and verify the remote branch SHA. Do not commit the
original supplied challenge documents or prompt/conversation inputs unless the
user explicitly asks for that publication.
