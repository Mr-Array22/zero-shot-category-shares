# 📊 Accuracy vs. Dashboard Bias: a Zero-Shot Classifier on Real CFPB Complaints

Code, frozen data and complete results for a study of how a zero-shot classifier can be accurate per message yet distort category shares on a dashboard.

A hosted zero-shot classifier (Jev 1.14, given only one-line category definitions) came within two accuracy points of a TF-IDF model trained on 12,000 labels, yet overstated one category on the dashboard by 7.75 percentage points. This repository traces the bias to one ambiguous definition and tests two fixes on 2,000 unseen complaints with a pre-registered decision rule.

## 🔑 Key results

| | Accuracy | Worst category-share error |
|---|---:|---:|
| **Original test** (Oct–Dec 2016, n = 2,000) | | |
| TF-IDF + logistic regression | 88.25% | 2.40 pp |
| Jev v1 (original definitions) | 86.25% | 7.75 pp |
| Jev v1 + share correction (ACC, fitted on dev) | — | 1.71 pp |
| **Fresh test** (1 Jan – 22 Apr 2017, n = 2,000, pre-registered) | | |
| TF-IDF + logistic regression | 86.40% | 2.05 pp |
| Jev v1 (original definitions) | 84.95% | 6.30 pp |
| Jev v1 + share correction | — | 0.79 pp |
| **Jev v2 (clearer definitions, frozen before scoring)** | **87.85%** | **0.70 pp** |

Full numbers, intervals and caveats are in [`results/`](results/README.md).

## 🗂️ Repository layout

```text
.
├── src/jevstudy/          # Installable package: all experiment code
│   ├── study.py           #   labels, request builder, response validation, hashing helpers
│   ├── prepare.py         #   build the frozen study (and the fresh 2017 set) from the Kaggle archive
│   ├── baselines.py       #   keyword rules + TF-IDF / logistic regression
│   ├── embed_baseline.py  #   MiniLM sentence embeddings + logistic regression
│   ├── jev_runner.py      #   budgeted, resumable, journaled Jev API calls
│   ├── metrics.py         #   accuracy, macro F1, share error, Brier, bootstrap, review policy
│   ├── report.py          #   metrics.json, comparison.csv, review policy, error list
│   ├── figures.py         #   standard result charts
│   ├── run.py             #   one-command pipeline (baselines → optional Jev → report → charts)
│   ├── diagnostics.py     #   McNemar tests, misroute analysis, reliability details
│   ├── quantify.py        #   share corrections: CC, ACC, PCC, PACC, EM
│   ├── revision.py        #   definitions v2: dev evaluation, freeze, fresh test, repeat check
│   ├── fresh_extras.py    #   exploratory fresh-set comparisons
│   └── paths.py           #   project locations
├── tests/                 # 62 unit tests (no network, no API key)
├── definitions/           # v1 and v2 category definitions (+ frozen copy, rationale)
├── data/
│   ├── frozen/            # frozen 15,000-complaint study and request files (inputs)
│   └── raw/               # Kaggle archive, downloaded on demand (git-ignored)
├── results/               # published evidence, one folder per experiment (see results/README.md)
│   ├── main/              #   original 2016 run: baselines + Jev v1 on dev/test
│   ├── dev-v2/            #   Jev v2 definitions on the development split
│   ├── fresh-2017/        #   pre-registered fresh test: Jev v1 vs v2, TF-IDF, keyword
│   └── repeat/            #   run-to-run stability check (200 complaints × 2)
├── docs/                  # pre-registration and data provenance
├── requirements/          # lock.txt (pinned), base.txt, embed.txt
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## ⚙️ Setup

Python **3.12** is required. The pinned NumPy/SciPy versions have no wheels for 3.14 and fall back to a source build that needs a Fortran compiler.

```bash
conda create -n jev python=3.12 -y && conda activate jev   # or: python3.12 -m venv .venv
python -m pip install -r requirements/lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests                         # expect: Ran 62 tests ... OK
```

## 🔁 Reproduce the results without API calls

Every command below reads saved responses and makes **zero model calls**. The analysis commands rewrite their output files inside `results/` with identical numbers.

```bash
python -m jevstudy.run --run runs/offline                    # baselines + metrics from data/frozen
python -m jevstudy.diagnostics --run results/main
python -m jevstudy.quantify --run results/main
python -m jevstudy.revision eval-dev --v1 results/main --v2 results/dev-v2
python -m jevstudy.revision eval-fresh --fresh results/fresh-2017 --train-source results/main
python -m jevstudy.revision eval-repeat --repeat results/repeat --source results/main
python -m jevstudy.fresh_extras
```

Embeddings baseline (downloads the model from Hugging Face):

```bash
python -m pip install -r requirements/embed.txt
python -m jevstudy.embed_baseline --run results/main --model sentence-transformers/all-MiniLM-L6-v2
```

## 💳 Re-run the Jev calls (paid)

Put your key in a local `.env` file (git-ignored; see `.env.example`) or the `TYPESAFE_API_KEY` environment variable. The key is never printed or saved. All runs are budget-capped, journaled and resumable; repeat a command with the same folder to resume without paying twice.

```bash
python -m jevstudy.run --run runs/my-run --execute-jev --budget-usd 1                  # ~3,000 calls
python -m jevstudy.prepare --fresh 2017q1 --exclude-study results/main/study.jsonl --fresh-n 2000 --out runs/fresh
```

The full pre-registered protocol (dev evaluation → freeze → fresh test) is documented in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) and implemented in `jevstudy.revision`. All 8,400 successful calls in this study cost about $0.29 at the documented list price ($0.042 per million input tokens; re-check before running).

## 📄 Data and license

- **Data:** CFPB Consumer Complaint Database (U.S. Government Works), via the Kaggle snapshot by Selene Reyes, version 1. Reference labels are the product each consumer selected; the database is not a statistical sample of consumers. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).
- **Code:** MIT; see [`LICENSE`](LICENSE) for the scope of the license.
- **Citation:** see [`CITATION.cff`](CITATION.cff).
