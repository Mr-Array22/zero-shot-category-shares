# Preregistration: Jev definitions v2 for the debt-collection / credit-reporting boundary

Status: drafted 2026-10-23, before any fresh-set request is built or any fresh-set result is seen. The v2 definitions are not frozen until `python revision.py freeze` writes the hash block below.

## Hypothesis

With the v1 definitions (`definitions/v1.json`), Jev over-assigns the `credit_reporting` category. On the development split the signed share error is about +6 pp, driven by collector-tradeline complaints (`debt_collection` predicted as `credit_reporting`) and card-issuer reporting complaints (`credit_card` predicted as `credit_reporting`). Rewriting three category definitions (`definitions/v2.json`; rationale in `definitions/v2_rationale.md`) reduces this aggregate bias without a material loss of accuracy on fresh, unseen complaints.

## Data

- Source: CFPB consumer complaint database, Kaggle `selener/consumer-complaint-database` version 1, pinned in `prepare.py`.
- Fresh window (Option A, verified): 2017-01-01 through 2017-04-23 inclusive. The CFPB changed its product taxonomy effective 24 April 2017. Complaints submitted before that date keep their original product names. Sources:
  - https://files.consumerfinance.gov/f/documents/201704_cfpb_Summary_of_Product_and_Sub-product_Changes.pdf (effective date and rule on previously submitted complaints)
- Option B (unused 2016 Q4 records) was not needed, because Option A could be verified. It is implemented as a fallback only.
- Exclusions: every complaint ID and every normalised 4,000-character text hash that appears in the original study (train, dev or test).
- Sampling: deterministic, label-blind, seed tag `fresh-2017q1-20261023`. Same normalisation, 4,000-character cap, duplicate handling and conflicting-label removal as the original study.
- Size: 2,000 fresh records. Each record is sent once per arm.

## Arms

| Arm | Definitions | Requests |
|---|---|---|
| Jev v1 | `definitions/v1.json` (identical to the original study) | 2,000 |
| Jev v2 | `definitions/v2.json` (frozen) | 2,000 |
| Keyword baseline | Fixed rule list in `study.py` | local |
| TF-IDF baseline | Refit on the original training split with the `baselines.py` settings; refit is checked to reproduce the stored dev and test predictions exactly before use | local |

Only the narrative text is sent to Jev. Model pinned to `jev-1.14.0`. Automatic retries are off.

## Primary metrics

1. Paired difference in maximum absolute share error, v2 minus v1, on the fresh set. Share error for a category is (predicted share minus true share) in percentage points.
2. Signed share error of `credit_reporting` for v2 on the fresh set.

## Decision rule (fixed before fresh results)

- Criterion A: the v2 `credit_reporting` signed share error has absolute value below 3.0 pp (point estimate).
- Criterion B: v2 accuracy minus v1 accuracy is at least -1.0 pp (point estimate).
- v2 counts as fixing the bias if and only if A and B both hold.
- The paired difference in maximum absolute share error (metric 1) is reported with a 95% paired bootstrap interval. A claim that v2 reduces aggregate share error requires that interval's upper bound to be below zero. This does not change the A/B rule.
- If the criteria are not met, v1 remains the production definition. The result is reported as is.

## Secondary metrics

- Accuracy and macro F1, with paired v2 minus v1 differences and 95% intervals.
- Per-category share errors for both Jev arms.
- Count of `debt_collection` predicted as `credit_reporting`, and of `credit_card` predicted as `credit_reporting`.
- Keyword and TF-IDF results, for context only.

## Statistics

- Percentile bootstrap, 500 resamples, seed 20261022. Each resample uses the same record indices for every arm, so differences are paired.
- Intervals are 2.5th and 97.5th percentiles of the resampled values.
- No multiple-comparison correction is applied. Secondary metrics are descriptive.

## Repeat-stability check

A separate check of 200 original test-split records, each sent twice more to v1 (400 requests), measures label agreement and probability differences against the original responses. It tests run-to-run stability only. It does not evaluate v2 and is not part of the decision rule.

## Already-seen data disclosure

- Test-split errors from the original run were inspected by the team before v2 was drafted. This is why the confirmatory evaluation uses the fresh 2017 Q1 set rather than the test split.
- The v2 draft was built only from the 1,000 development records and their Jev v1 responses. Development results are therefore not confirmatory. Dev metrics written by `eval-dev` are labelled as such.
- The fresh set was built label-blind and has not been inspected for errors before the decision rule was fixed.

## Frozen definitions

<!-- FROZEN:BEGIN -->
definitions_file: definitions/v2.json
definitions_sha256: add74c3548048d2a759fad4702a3ea43f8fd1f2b6c120d90f06eadfe56cfc482
frozen_utc: 2026-10-23T11:58:11.748264+00:00
<!-- FROZEN:END -->

Freezing copies `definitions/v2.json` to `definitions/v2.frozen.json`. `build-fresh`, `eval-fresh` and `run` on prepared fresh arms refuse to proceed if `definitions/v2.json` no longer matches the hash in this block.

## Planned runs

Token estimate: about 780 input tokens per request, list price $0.042 per million input tokens. Budget reservation per call is $0.00275, and the per-run budget default is $1.

| Run | Requests | Estimated list price |
|---|---|---|
| Development v2 (`runs/dev-v2`) | 1,000 | $0.033 |
| Fresh v1 and v2 (`runs/fresh/jev-v1`, `runs/fresh/jev-v2`) | 4,000 | $0.131 |
| Repeat stability (`runs/repeat`) | 400 | $0.013 |
| Total | 5,400 | $0.177 |

These are planning estimates from list price. The provider invoice is the source of truth.

## Execution order

1. `python prepare.py --fresh 2017q1 --exclude-study runsmy-run/study.jsonl --fresh-n 2000 --out runs/fresh` (downloads the archive if absent)
2. `python revision.py build-dev-v2 --source runsmy-run --out runs/dev-v2`
3. `python revision.py run --folder runs/dev-v2 --execute-jev --budget-usd 1`
4. `python revision.py eval-dev --v1 runsmy-run --v2 runs/dev-v2`
5. `python revision.py freeze --definitions definitions/v2.json` (only after step 4, and only if v2 is final)
6. `python revision.py build-fresh --fresh runs/fresh`
7. `python revision.py run --folder runs/fresh/jev-v1 --execute-jev --budget-usd 1`
8. `python revision.py run --folder runs/fresh/jev-v2 --execute-jev --budget-usd 1`
9. `python revision.py eval-fresh --fresh runs/fresh`
10. `python revision.py build-repeat --source runsmy-run --n 200 --copies 2 --out runs/repeat`
11. `python revision.py run --folder runs/repeat --execute-jev --budget-usd 1`
12. `python revision.py eval-repeat --repeat runs/repeat --source runsmy-run`

Steps 3, 7, 8 and 11 are the only billable steps. Each stops on the first failed call with no automatic retry. Resume with `--retry-failed` only after inspecting the journal.
