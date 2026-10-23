# 📁 Results

Each folder holds the complete evidence for one experiment: inputs (`study.jsonl`, `requests.jsonl`, `data-manifest.json` with SHA-256 hashes), raw Jev responses and call journals (no API key), and computed metrics.

| Folder | What it is | Key files |
|---|---|---|
| `main/` | Original run on the 2016 study: keyword, TF-IDF, embeddings and Jev v1 on development (1,000) and test (2,000) | `metrics.json`, `comparison.csv`, `diagnostics.json`, `quantification.json`, `embed-metrics.json`, `figures/` |
| `dev-v2/` | Jev with the v2 definitions on the development split (design data, not confirmatory) | `dev-metrics.json` |
| `fresh-2017/` | Pre-registered test on 2,000 unseen complaints (1 Jan – 22 Apr 2017): Jev v1 and v2 arms, TF-IDF, keyword | `fresh-metrics.json` (pre-registered), `fresh-extras.json` (exploratory) |
| `repeat/` | 200 test complaints sent twice more to Jev v1 to measure run-to-run stability | `repeat-metrics.json` |

## 🔀 Folder names used during the experiment

The repository was reorganized after all runs finished. Paths recorded inside some JSON files and in `docs/PREREGISTRATION.md` use the original names:

| Original path | Current path |
|---|---|
| `runsmy-run/` | `results/main/` |
| `runs/dev-v2/` | `results/dev-v2/` |
| `runs/fresh/` | `results/fresh-2017/` |
| `runs/repeat/` | `results/repeat/` |
| `reference-run/{study,requests}.jsonl`, `data-manifest.json` | `data/frozen/` |
| root-level `*.py` scripts | `src/jevstudy/` (run as `python -m jevstudy.<module>`) |

File contents were not changed by the move. Every file's SHA-256 was checked before and after, and every analysis was re-run from the new layout with identical numbers.
