# Definitions v2: rationale (dev split only)

Scope: the 1,000 development records in `runsmy-run/study.jsonl` and Jev v1 responses for those records. No test-split text or test error was read for this draft. Complaint IDs are cited instead of quotes.

## Dev evidence

Jev v1 on dev: accuracy 0.875 (875/1000). Largest off-diagonal cells:

| True label | Jev v1 prediction | Count |
|---|---|---|
| debt_collection | credit_reporting | 34 |
| credit_card | credit_reporting | 28 |
| mortgage | debt_collection | 10 |
| credit_reporting | debt_collection | 5 |

Credit reporting therefore absorbs 62 net extra predictions on dev (+6.2 pp share error). Mortgage to debt_collection (10) is noted but left unchanged because the dev evidence does not show a consistent definitional cause.

### debt_collection to credit_reporting (34 IDs)

IDs: 1998438, 2006025, 2017816, 2018334, 2018383, 2026170, 2027610, 2034790, 2042270, 2050059, 2055677, 2059148, 2062945, 2064886, 2073214, 2074480, 2084376, 2086814, 2087133, 2091058, 2103934, 2112764, 2116648, 2116779, 2117568, 2117604, 2117835, 2122311, 2127706, 2127721, 2130260, 2130452, 2130554, 2131676.

- 23 of 34 contain a collector or collection-related cue (collector, collection agency, validation or verification of a debt). Typical grievance: a collector's collection tradeline on the report, with requests to verify, remove or correct it.
- Only 8 of 34 name a bureau. The v1 model therefore keys on "credit report" wording rather than on the collector.

### credit_card to credit_reporting (28 IDs)

IDs: 1996849, 2001083, 2007912, 2009313, 2012488, 2021433, 2022066, 2023220, 2027967, 2031531, 2032203, 2037763, 2038091, 2038546, 2040309, 2052912, 2052968, 2060195, 2069618, 2071469, 2077422, 2095157, 2095511, 2104357, 2107269, 2132905, 2137722, 2138668.

- Most are a card issuer's or lender's reporting of late payments, balances or account status on a card account. The v1 credit_card text excludes "credit-report accuracy as the primary complaint", which plausibly pushes these to credit_reporting. This is a hypothesis to be tested on fresh data, not a proven cause.
- 10 of 28 name a bureau, and 5 contain a collector cue.

### Counter-evidence considered

Among dev rows that contain both a bureau term and a collection cue, the gold labels are split: 28 credit_reporting and 28 debt_collection. Jev v1 gets 27 of the credit_reporting rows and 22 of the debt_collection rows right. A bureau cue alone is therefore not a clean discriminator. The v2 wording uses the main grievance (who the complaint is against and what it asks for) instead of a keyword.

## Changes in v2

- debt_collection: names the collector's conduct and the main ask (remove, verify or correct a collection account reported to a bureau). Targets the 23 collection-cue errors.
- credit_reporting: restricted to bureau handling as the main grievance, and explicitly excludes complaints against a collector and a lender's reporting of its own account.
- credit_card: explicitly includes a card issuer's reporting on its own card account, and keeps bureau handling out of the category. Targets the 28 card-to-credit_reporting errors.
- bank_account, mortgage and the instructions text are unchanged.

## Risks

- These changes were drafted by looking at dev errors. Fresh evaluation is the only test of whether they generalise.
- The credit_card change may shift errors toward credit_card for true credit_reporting complaints. The fresh set must check this (credit_reporting share error and accuracy).
- The debt_collection wording relies on "main ask". Jev may read this differently from the team's intent, which is why the decision rule in PREREGISTRATION.md uses share error and accuracy together.
