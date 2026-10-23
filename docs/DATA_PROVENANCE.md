# Real-data provenance and reuse

**Kaggle dataset:** Selene Reyes, *Consumer Complaint Database*.

[Dataset card](https://www.kaggle.com/datasets/selener/consumer-complaint-database)

**Pinned download:** `https://www.kaggle.com/api/v1/datasets/download/selener/consumer-complaint-database?datasetVersionNumber=1`

The archive retrieved on 22 October 2026 contains `rows.csv`. The full downloaded ZIP is 184,119,866 bytes and the uncompressed CSV member is 737,001,477 bytes. Its SHA-256, member CRC and frozen-study hashes are recorded in `reference-run/data-manifest.json`. The current card can describe a later file size; do not use its current preview as proof that another version matches version 1.

**Original publisher:** Consumer Financial Protection Bureau (CFPB).

[Official database and reuse statement](https://www.consumerfinance.gov/data-research/consumer-complaints/), [official field reference](https://cfpb.github.io/api/ccdb/fields.html)

The Kaggle card identifies the material as U.S. Government Works and traces it to CFPB/data.gov. CFPB states that its published complaint data are freely available for anyone to use, analyze and build on. The study retains attribution to both the original publisher and the Kaggle host. The Python code's MIT license does not relicense complaint narratives or third-party sources.

The frozen study is a derived subset of actual records. Changes: select five original product labels; select dates before 2017; normalize whitespace; discard empty/very short text; cap model input at 4,000 characters; deduplicate exact lowercased model-input groups by earliest date; remove conflicting-label groups; sample each time period deterministically without class-balancing quotas. No complaints or reference labels were generated with AI.

The selected subsets contain 12,000 training, 1,000 development and 2,000 test narratives. Only the narrative is used for inference. Complaint IDs and dates support traceability/splitting; Product is the reference target. The source's other fields are not included in the frozen subset or any API payload.

CFPB labels describe the product the consumer selected. The database is not representative of all consumers' experiences. Public complaints are not findings of fact about the companies or products involved. This study examines classification and aggregation, not financial advice or company rankings.

The reference package does not redistribute the full Kaggle archive or its cover image. It includes the attributed study subset needed to reproduce the experiment and a downloader for the pinned archive. Review any narrative excerpts before quoting them publicly.
