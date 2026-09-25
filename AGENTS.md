# Amazon ML Challenge 2026

- Read README.md, both PDFs under `Statement and Docs`, and the official validator.
- Attached conversation.md is historical advice, not evidence of current state.
- Use only the supplied competition data for business matching. Technical research
  is allowed; external business identity lookup, geocoding and enrichment are not.
- Preserve source TSVs and utils/validate_submission.py. Always use tab separators.
- Final model must meet MIT/Apache-2.0 and <=8B parameter requirements. Verify any
  checkpoint license separately from its code library license.
- Country labels are an open set. Do not exclude France or map it to India/US.
- Macro F0.5 is per S1, with true/predicted empty = 1, false singleton merge = 0.
- Multiple matches per S1 are valid. Do not impose one-to-one S1-to-target matching.
- Split by S1/connected identity group, never by random candidate pairs. Preserve
  full negative retrieval difficulty when reporting competitive validation scores.
- Use seed 42. Record parameters, candidate population and runtime with results.
- Never call a reduced-corpus pilot full-corpus recall or a leaderboard estimate.
- On this laptop use disk-backed operations and chunked sparse retrieval. No dense
  all-pairs matrices. Keep caches/models/outputs out of Git.
- Keep the exact candidates scored by the final model for candidate_pairs.tsv.
- Run metric/retrieval tests and official validator with --check-ids before delivery.
- No leaderboard submission or remote publication is part of the initial audit task.
