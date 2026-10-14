# 🌀 Digo — NLP Disaster Tweet Classification

**Project 7 | Data Science & AI Internship**

**Prepared by:** Bhawna Chouhan

A complete Natural Language Processing pipeline that classifies tweets as **disaster-related** or **non-disaster**, with a deployed Streamlit application.

### 🚀 [**Launch the live app →**](https://project7digo-6nxbuqxvpnitprohz9cjts.streamlit.app/)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://project7digo-6nxbuqxvpnitprohz9cjts.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

---

## Table of contents

- [Overview](#overview)
- [Problem statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Results](#results)
- [Key findings](#key-findings)
- [Limitations](#limitations)
- [Tech stack](#tech-stack)
- [Credits](#credits)

---

## Overview

During an emergency, social media is often faster than official channels. Disaster-relief
agencies and newsrooms would like to monitor it automatically — but the vocabulary of
disaster is also the vocabulary of everyday exaggeration.

| Tweet | Contains | Real disaster? |
|---|---|---|
| "Forest fire near La Ronge Sask. Canada" | *fire* | ✅ **Yes** |
| "that exam absolutely destroyed me" | *destroyed* | ❌ No |
| "My new song is FIRE 🔥" | *fire* | ❌ No |
| "13,000 people receive wildfires evacuation orders" | *wildfires* | ✅ **Yes** |

A system that searches for the word *fire* flags all four. The real task is **contextual
understanding**, which is what makes this a genuine NLP problem rather than a lookup table.

---

## Problem statement

Build a machine learning model that predicts whether a tweet refers to a real disaster
(`1`) or not (`0`), handling figurative, metaphorical and sarcastic uses of
disaster-related words.

---

## Objectives

1. Audit the supplied dataset and document every data-quality issue found.
2. Explore the data to establish what genuinely separates the two classes.
3. Build a reusable, tested preprocessing pipeline that **preserves meaning**.
4. Engineer numeric features and implement real sentiment analysis.
5. Compare multiple models fairly on an untouched test set.
6. Investigate and fine-tune DistilBERT as an advanced comparison.
7. Select a final model on evidence, save it, and verify it reloads.
8. Deploy a Streamlit application that uses the saved model.

---

## Dataset

**File:** `data/twitter_disaster.csv` — the dataset supplied with the project brief
(the Kaggle *"Natural Language Processing with Disaster Tweets"* training split).

### ⚠️ Dataset size — an honest note

> The written project statement describes **"10,000 hand-classified tweets"**.
> The dataset actually supplied contains **7,613 rows**.
>
> **This project uses the supplied 7,613 rows and nothing else.** No rows were
> synthesised, duplicated or imported from elsewhere to reach 10,000. Every statistic,
> chart, split and metric in this repository is derived from those 7,613 real rows.
> Mentor guidance was explicit that the actual supplied dataset must be used.

### Verified audit

All values below are computed from the file in notebook Section 3 — none are copied
from the brief.

| Check | Verified value |
|---|---|
| Shape | **7,613 rows × 5 columns** |
| Columns | `id`, `keyword`, `location`, `text`, `target` |
| Missing `keyword` | 61 (0.80%) |
| Missing `location` | 2,534 (33.28%) |
| Missing / empty `text` | 0 |
| Duplicate `id` | 0 |
| Duplicate `text` | 110 |
| **Texts with contradictory labels** | **18** (same text labelled both 0 and 1) |
| Distinct keywords | 221 (URL-encoded, e.g. `body%20bags`) |
| Distinct locations | 3,293 (unvalidated free text) |
| Tweet length | 7–157 characters, 1–31 words |

### Class distribution

| Label | Meaning | Count | Share |
|---|---|---|---|
| `0` | Non-disaster | 4,342 | 57.03% |
| `1` | Disaster-related | 3,271 | 42.97% |

Mild imbalance. A model that always predicts "non-disaster" scores **57.0% accuracy**
while finding zero real disasters — which is why F1 and recall are reported throughout,
never accuracy alone.

### Rows used for modelling

| Stage | Rows |
|---|---|
| Supplied file | 7,613 |
| After removing 18 contradictory texts | 7,577 |
| After removing duplicate `(text, target)` pairs | **7,485** |
| Training set (80%) | 5,988 |
| Test set (20%, untouched until final evaluation) | 1,497 |

Deduplication happens **before** the split. Had a duplicated tweet landed in both
training and test, the model would simply memorise it and the test score would be
dishonestly high.

---

## Project structure

```
Project_7_Digo/
|-- app.py                      # Streamlit application
|-- requirements.txt            # runtime dependencies
|-- models/
|   |-- selected_model.joblib   # trained pipeline (1.05 MB)
|   `-- model_metadata.json     # metrics + config the app reads
|-- src/
|   |-- config.py               # paths, seed, labels
|   |-- preprocessing.py        # text cleaning (negation-preserving)
|   |-- features.py             # 13 engineered features
|   |-- sentiment.py            # VADER + TextBlob
|   `-- logger.py               # optional prediction logging
`-- .streamlit/config.toml      # theme
```

> `src/preprocessing.py` is required at runtime — the saved model references its
> `clean_text` function, so the pipeline cannot be unpickled without it.

**Kept locally, not in this repository:** the Jupyter notebook, dataset, tests, the
full set of 20 charts, and the presentation. They are excluded via `.gitignore` so the
deployment stays small. To publish any of them: `git add -f <path>`.

---

## Quick start

```bash
git clone https://github.com/bhawnachouhan01/Project_7_Digo.git
cd Project_7_Digo
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>. Requires **Python 3.12**.

---

## How it works

```
tweet -> clean text -> TF-IDF -> calibrated LinearSVC -> label + probability
```

1. **Preprocessing** — HTML entities unescaped, URLs become `urltoken`, `@mentions`
   become `usertoken`, hashtag words are kept (`#CaliforniaWildfire` ->
   *california wildfire*), numbers are kept. **Negation words are preserved.**
2. **Vectorization** — TF-IDF over the cleaned text
   (30,000 terms, unigrams + bigrams).
3. **Classification** — Linear SVC wrapped in probability calibration, so the
   confidence shown in the app is a real probability.

### Why negation is preserved

Standard cleaning deletes *no* and *not*, which inverts the meaning:

| | Result |
|---|---|
| Standard cleaning | `"There is no flooding"` -> `flooding` (wrong) |
| This project | `"There is no flooding"` -> `no flooding` (correct) |

---

## Results

Measured on **1,497 held-out tweets** never seen during training.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| DistilBERT (fine-tuned) <- best | 0.830 | 0.838 | 0.746 | **0.789** |
| Calibrated LinearSVC (tuned) **<- deployed** | 0.788 | 0.770 | 0.718 | **0.743** |
| Logistic Regression (TF-IDF) | 0.788 | 0.797 | 0.676 | **0.731** |
| Naive Bayes (TF-IDF) | 0.792 | 0.843 | 0.630 | **0.721** |
| Random Forest (TF-IDF + SVD) | 0.758 | 0.800 | 0.575 | **0.669** |
| Baseline (majority class) | 0.574 | 0.000 | 0.000 | **0.000** |

![Model comparison](reports/figures/16_comparison_with_distilbert.png)

### Deployed model

| Metric | Value |
|---|---|
| Accuracy | **78.82%** |
| F1 (disaster class) | **0.7429** |
| ROC-AUC | 0.8421 |
| Artifact size | 1.05 MB |
| Inference | 0.13 ms per tweet |

Of **638 real disasters** in the test set the model found **458**
and missed **180**. Of its **595 alerts**,
**77.0%** were correct.

![Confusion matrix](reports/figures/17_confusion_matrix.png)

---

## Key findings

1. **Keyword matching cannot solve this.** Hundreds of non-disaster tweets contain
   *fire*, *destroyed* or *crash*.
2. **DistilBERT is the better model** (0.789 vs 0.743 F1) but is
   ~255 MB, over GitHub's 100 MB file limit, so the
   1.05 MB classical pipeline is deployed instead. A documented
   trade-off, not a claim that the two are equivalent.
3. **Negation defeats both models.** *"There is no flooding in our area"* is classified
   as a disaster by the deployed model **and** by DistilBERT. Preserving the word *no*
   was necessary but not sufficient — the training data contains too few negated
   disaster statements for either model to learn the pattern.
4. **Engineered features and sentiment added little** over the text alone, and
   preprocessing variants moved the score less than cross-validation noise. Both are
   honest negative results.
5. **Human annotators disagreed on 18 tweets**, so some error is irreducible. A model
   scoring ~100% here would indicate data leakage, not skill.

---

## Limitations

- **Not an emergency service.** It classifies *language*; it cannot verify that an
  event actually happened.
- **Negated statements are misclassified** (see Key findings).
- **English only**, trained on 2015-era tweets.
- **Sarcasm and very short tweets** are unreliable.
- **No disaster-type detection** — the dataset provides no reliable type label.

---

## Tech stack

Python 3.12 · scikit-learn · pandas · NumPy · NLTK · VADER · TextBlob · Streamlit

*Development also used gensim, PyTorch and Hugging Face Transformers for the Word2Vec
and DistilBERT experiments.*

---

## Credits

**Dataset** — [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started),
Kaggle. Supplied with the project brief as `twitter_disaster.csv`.

**References**
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* — <https://arxiv.org/abs/1910.01108>
- Hutto & Gilbert (2014), *VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis of Social Media Text*, ICWSM-14

**Author** — Bhawna Chouhan · Project 7 · Data Science & AI Internship
