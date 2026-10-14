# Digo - NLP Disaster Tweet Classification
## Mentor Presentation Plan (15 slides)

> **Auto-generated from real results.** Every figure below is read from
> `models/model_metadata.json` and `reports/model_comparison.csv`, which the
> notebook wrote from actual evaluation runs. Regenerate with
> `.venv\Scripts\python.exe make_presentation.py`.
>
> A ready-to-present deck is also produced: **`Project_7_Digo_Presentation.pptx`**.

**Final model:** Calibrated LinearSVC (tuned)
**Headline result:** 78.82% accuracy, 0.7429 F1 (disaster class),
ROC-AUC 0.8421, measured on 1,497 held-out tweets.

---

## Slide 1 - Title

**Title:** Digo - NLP Disaster Tweet Classification
**Subtitle:** Can a machine tell a real emergency from a figure of speech?

**Content:** Project 7, Data Science & AI Internship. Final model
Calibrated LinearSVC (tuned) reaching 78.8% accuracy and 0.743 F1.

**Visual:** Title slide, no chart.

---

## Slide 2 - The Problem & Objectives

**Title:** The Problem & Objectives - why keyword matching fails

**Content:**
- Twitter is often faster than official channels during an emergency.
- Disaster vocabulary is also everyday exaggeration vocabulary:
  - *"Forest fire near La Ronge Sask. Canada"* -> REAL disaster
  - *"that exam absolutely destroyed me"* -> NOT a disaster
  - *"My new song is FIRE"* -> NOT a disaster
- Searching for "fire" flags all three. The task is **contextual understanding**,
  not keyword lookup.
- Objective: classify 1 = disaster-related, 0 = non-disaster, judged on operational
  metrics because the two error types differ:
  - False negative = a real emergency missed (the dangerous error).
  - False positive = a false alarm (causes alarm fatigue).
- Full lifecycle: Audit -> EDA -> Preprocessing -> Keywords -> Features -> Sentiment
  -> Vectorization -> Leakage-safe split -> Models -> DistilBERT -> Evaluation -> App.

**Visual:** The three-example contrast table.

---

## Slide 3 - Dataset & Data-Quality Audit

**Title:** Dataset & Data-Quality Audit

**Actual results to insert:**

| Check | Value | Why it matters |
|---|---|---|
| Rows x columns | 7,613 x 5 | Brief claimed 10,000 - the supplied file is what we use |
| Missing `keyword` | 61 (0.8%) | Nearly complete; curated metadata |
| Missing `location` | 2,534 (33.3%) | Unvalidated free text - excluded from the model |
| Duplicate IDs | 0 | Row identity is clean |
| Duplicate tweet text | 110 | Removed **before** splitting to prevent leakage |
| Contradictory labels | 18 texts | Humans disagreed - a realistic accuracy ceiling |
| Non-disaster (0) | 4,342 (57.0%) | Majority-class baseline = 57% accuracy |
| Disaster (1) | 3,271 (43.0%) | Mild imbalance - accuracy alone misleads |

**Key point to state aloud:** the project statement describes 10,000 tweets; the
supplied dataset contains 7,613. The supplied data was used
as-is - no rows were synthesised. After removing duplicates and contradictions,
**7,485 rows** were modelled.

**Visual:** The audit table.

---

## Slide 4 - Keyword Analysis

**Title:** Keyword Analysis - the curated keyword separates the classes, but not the
way you would expect

**Content:**
- The `keyword` column separates classes strongly - roughly 8% to 95% disaster rate.
- Crucially the pattern is **not** "dramatic word = disaster": *blazing*, *ruin* and
  *aftershock* are mostly figurative, while *wreckage* and *derailment* are almost
  always literal.
- Excluded from the model anyway - it is dataset metadata a live user never supplies.
- Class balance (57% / 43%) means a "never a disaster" model scores 57% accuracy while
  finding zero real disasters - so F1 and recall are reported throughout.

**Visual:** `reports/figures/07_keyword_analysis.png`
(class distribution if needed: `reports/figures/01_class_distribution.png`)

---

## Slide 5 - Frequency Is Not Discrimination

**Title:** Frequency Is Not Discrimination - a common word is not necessarily a useful one

**Content:**
- A frequent word is not necessarily a useful one. Hundreds of non-disaster tweets
  contain *fire*, *destroyed* or *crash* - the concrete reason keyword matching
  cannot solve this task.
- *wreck* and *destroyed* sit near the ambiguous middle, because those are the words
  English speakers most often use figuratively.

**Visual:** `reports/figures/10_word_discrimination.png`

---

## Slide 6 - Text Preprocessing

**Title:** Text Preprocessing - cleaning that preserves meaning

**Content:**
- Preprocessing **transforms** text; feature engineering **measures** it.
- URLs -> `urltoken`, @mentions -> `usertoken` (the fact of a link is signal).
- Hashtag words kept and CamelCase split: `#CaliforniaWildfire` -> *california wildfire*.
- Numbers kept: *"13,000 evacuated"*, *"M5.8 earthquake"*.
- **Negation deliberately preserved** - the most deliberate decision here:
  - standard cleaning: *"There is no flooding"* -> `flooding` (meaning inverted)
  - our cleaning: *"There is no flooding"* -> `no flooding` (correct)
- Necessary - but **not sufficient**. Slide 12 shows the model still misreads it.
- One implementation, imported by both notebook and app - they cannot drift apart.

**Visual:** The before/after negation comparison.

---

## Slide 7 - Feature Engineering

**Title:** Feature Engineering - 13 numeric features from the raw tweet

**Content:** Strongest signals are `url_count` (disaster tweets link to news) and
`mention_count` (non-disaster tweets are conversational). Even these are only
small effects - an honest finding. `keyword` and `location` were **deliberately
excluded**: they are dataset metadata that a live user would not supply, so
training on them would inflate scores and then fail in deployment.

**Visual:** `reports/figures/11_feature_effect_sizes.png`

---

## Slide 8 - Sentiment Analysis

**Title:** Sentiment Analysis - implemented and measured

**Content:** VADER (built for social media, handles negation) plus TextBlob
(adds subjectivity). Disaster tweets are more negative and more objective. But
sentiment answers a **different question**: a factual disaster report reads
neutral, while *"this traffic is killing me"* reads strongly negative. A
sentiment-only rule performs **worse than guessing the majority class**, so it is
used as a supporting signal only.

**Visual:** `reports/figures/13_sentiment_analysis.png`

---

## Slide 9 - Vectorization & Embeddings

**Title:** Vectorization & Embeddings - five distinct concepts

| Approach | What it produces | Context-aware? | Used here |
|---|---|---|---|
| Tokenization | Text -> list of tokens | n/a | Yes |
| Bag of Words | Sparse term counts | No | Yes |
| TF-IDF | Counts weighted by rarity | No | **Yes - final model** |
| Word2Vec | 100-dim dense word vectors | No | Yes (trained on corpus) |
| DistilBERT | 768-dim contextual vectors | **Yes** | Yes - fine-tuned |

**Key points:** vectorization is *not* label encoding - words are not ordered
categories. Word2Vec was trained on our own corpus, avoiding an 822 MB GloVe
download. Static embeddings give *fire* one fixed vector, but our whole task is
that "forest fire" and "song is fire" differ - hence DistilBERT.

**Visual:** The comparison table.

---

## Slide 10 - Model Comparison

**Title:** Model Comparison - all models on the same untouched test set

**Actual results:**

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 |
|---|---|---|---|---|---|
| DistilBERT (fine-tuned) | 0.8303 | 0.8380 | 0.7461 | 0.7894 | 0.8237 |
| Calibrated LinearSVC (tuned) | 0.7882 | 0.7697 | 0.7179 | 0.7429 | 0.7814 |
| Linear SVC (tuned) | 0.7896 | 0.7770 | 0.7100 | 0.7420 | 0.7822 |
| LogReg + engineered features | 0.7882 | 0.7945 | 0.6787 | 0.7320 | 0.7785 |
| Logistic Regression (TF-IDF) | 0.7882 | 0.7967 | 0.6755 | 0.7311 | 0.7782 |
| LinearSVC + engineered features | 0.7876 | 0.7974 | 0.6724 | 0.7296 | 0.7773 |
| Linear SVC (TF-IDF + stemming) | 0.7836 | 0.7865 | 0.6755 | 0.7268 | 0.7738 |
| Naive Bayes (BoW counts) | 0.7909 | 0.8218 | 0.6505 | 0.7262 | 0.7785 |
| Linear SVC (TF-IDF) | 0.7816 | 0.7864 | 0.6693 | 0.7231 | 0.7714 |
| Naive Bayes (TF-IDF) | 0.7923 | 0.8428 | 0.6301 | 0.7211 | 0.7778 |
| SGD Classifier (TF-IDF) | 0.7662 | 0.7432 | 0.6897 | 0.7154 | 0.7585 |
| Random Forest (TF-IDF + SVD) | 0.7575 | 0.7996 | 0.5752 | 0.6691 | 0.7389 |
| KNN (TF-IDF + SVD) [demo] | 0.7448 | 0.7667 | 0.5768 | 0.6583 | 0.7273 |
| Baseline (majority class) | 0.5738 | 0.0000 | 0.0000 | 0.0000 | 0.3646 |

**Point to state:** linear models dominate on sparse text. Random Forest and KNN
underperform, confirming with evidence that tree and distance-based models are a
poor fit for high-dimensional text.

**Visual:** The table above, plus `reports/figures/16_comparison_with_distilbert.png`.
The classical models are statistically tied with each other; DistilBERT leads the field.

---

## Slide 11 - DistilBERT

**Title:** DistilBERT - actually fine-tuned

**Content:**
- 6 transformer layers, ~66M parameters, distilled from BERT-base.
- Self-attention makes each word's vector depend on its sentence - exactly what
  this task needs.
- Model card reviewed: intended for fine-tuning; documented risks include biased
  predictions and English-only support.
- A sentiment checkpoint would be the **wrong** choice - different question. We
  fine-tuned the **base** checkpoint on our own labels.
- Trained for real: 2 epochs, lr=3e-05,
  batch=32, max_len=64,
  on CPU - **15.0 minutes**.
- Result: accuracy **0.8303**, F1 **0.7894**,
  macro-F1 0.8237, ROC-AUC 0.8897.
- **Not deployed:** ~255 MB exceeds GitHub's 100 MB file limit.

**Visual:** Training log excerpt + the comparison bars.

---

## Slide 12 - Evaluation & Error Analysis

**Title:** Evaluation & Error Analysis

**Actual results:**
- Accuracy **0.7882**, Precision **0.7697**,
  Recall **0.7179**, F1 **0.7429**
- Macro-F1 0.7814, Weighted-F1 0.7871
- ROC-AUC 0.8421, Average precision 0.8266
- Confusion matrix: TN=722, FP=137,
  FN=180, TP=458

**Point to state:** of 638 real disasters in the test set, the model
found 458 (71.8%) and missed
180. Of its 595 alerts, 77.0% were correct.

**Visual:** `reports/figures/17_confusion_matrix.png` and
`reports/figures/18_roc_pr_curves.png`

---

## Slide 13 - The Negation Finding

**Title:** The Negation Finding - the most instructive result

**Actual results from the probe comparison (notebook Section 13.2.1):**

| Probe tweet | Deployed model | DistilBERT |
|---|---|---|
| "This traffic is killing me..." | WRONG (0.701) | **correct** (0.381) |
| "There is no flooding in our area..." | WRONG (0.854) | WRONG (**0.915**) |
| "The fire alarm was a false alarm..." | WRONG (0.819) | WRONG (0.550) |

Unambiguous probes overall: **TF-IDF 10/13, DistilBERT 11/13**. On the
figurative/negation subset: **3/6 vs 4/6**.

**Content:**
- Hypothesis going in: self-attention should let *no* modify *flooding*, fixing negation.
- **What DistilBERT did fix - figurative language.** "This traffic is killing me" goes
  from a confident wrong answer to a correct one.
- **What it did not fix - negation.** Both models still call "no flooding" a disaster,
  and DistilBERT does so *more* confidently.
- **Why: the training data, not the architecture.** Disaster words appear in ~3,000
  tweets, almost all genuine reports; negated statements are rare, so both models learn
  an overwhelming "flooding implies disaster" prior.
- Preprocessing **does** preserve the *no* - preserving the token was necessary but not
  sufficient.
- Reported as a real limitation in the notebook, the README and the app's own UI.

**Point to state aloud:** this is the result that did *not* go as predicted, and
reporting it that way is the honest thing to do.

**Visual:** The probe comparison table.

---

## Slide 14 - Final Model & Streamlit Application

**Title:** Final Model & Streamlit Application - chosen on evidence, then actually shipped

| Criterion | Value |
|---|---|
| Test F1 (disaster) | 0.7429 |
| Macro-F1 | 0.7814 |
| ROC-AUC | 0.8421 |
| Cross-validated F1 | 0.7611 +/- 0.0162 |
| Artifact size | 1.05 MB |
| Inference speed | 0.131 ms per tweet |
| Probability output | Calibrated |

**Selected:** Calibrated LinearSVC (tuned) - fits GitHub and Streamlit Cloud,
gives a real calibrated probability for the app, and is linear and inspectable.
DistilBERT was reported but not deployed
(~244x larger,
159x slower).

**The app:**
- Branded dark-theme UI with **About Me**, **About the Project** and
  **Project Overview** panels introducing the author and the problem.
- Paste a tweet -> colour-coded verdict (red = disaster, green = non-disaster)
  plus a calibrated confidence bar and a low-confidence warning near the boundary.
- Two result tabs: *How the model read this* (cleaned text + measured properties)
  and *Sentiment (separate)*, with the difference from the prediction explained.
- A **model-comparison panel** showing the deployed model against the fine-tuned
  DistilBERT, stating plainly that the transformer is better and excluded only on
  size and speed.
- Every figure read live from `model_metadata.json` - never hardcoded.
- Five one-click example tweets (clear / figurative / negation / ambiguous / chat).
- Input validation, friendly errors, no tracebacks, no debug output.
- Limitations stated in the UI, including the negation weakness.

**Demo tip:** click the **Negation** example on stage - the model gets it wrong, which
is the perfect lead-in to Slide 13.

**Visual:** The selection criteria table, plus a screenshot of the running app.

---

## Slide 15 - Conclusions, Limitations & Future Scope

**Title:** Conclusions, Limitations & Future Scope

**Content:**
- Delivered a working classifier at 78.8% accuracy /
  0.743 F1 on held-out data, plus a working Streamlit app (verified
  locally; deployment-ready but not publicly hosted).
- Honest findings worth stating:
  - Preprocessing choices moved the score less than cross-validation noise.
  - Engineered features and sentiment added little over text alone.
  - `keyword` was excluded despite being predictive - it would not exist live.
- Limitations: English only; 2015-era vocabulary; classifies language, not facts;
  no disaster-type label exists in the data; sarcasm remains hard.
- Human annotators disagreed on 18 texts - ~100% accuracy would indicate leakage.
- Future: modern labelled data, DistilBERT behind an API, explainability (SHAP),
  threshold tuned to an operational cost balance.

**Visual:** Summary slide.

---

## Presenting tips

- **Lead with the problem, not the model.** The "fire" contrast on Slide 2 makes
  the whole project make sense in 20 seconds.
- **Be ready for "why only 79%?"** The answer is on Slide 4: human
  annotators disagreed on 18 identical tweets, and tweets average ~15 words.
  Published results on this dataset cluster in the same range.
- **Be ready for "why not use `keyword`?"** Slide 9 - it is dataset metadata the
  app will never receive. Using it would benchmark well and fail in production.
- **Be ready for "why not deploy DistilBERT?"** Slide 14 - ~255 MB vs GitHub's
  100 MB file limit, and much slower inference for a modest gain.
