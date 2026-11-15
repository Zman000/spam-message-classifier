# SMS Spam and Scam Detection Using Supervised Machine Learning Algorithms

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-green.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)

A complete, reproducible, and explainable Machine Learning coursework project demonstrating supervised binary text classification for SMS spam and fraudulent message detection.

---

## 1. Project Introduction & Problem Statement

Short Message Service (SMS) remains a critical communication channel for banking alerts, authentication codes (OTPs), and personal correspondence. However, the absence of native cryptographic sender verification in legacy telecommunications protocols makes SMS a prime target for spam campaigns, smishing (SMS phishing), and financial lottery scams.

### Problem Statement
Given an incoming SMS text message $x$, predict a binary class label $y \in \{0, 1\}$:
- **Class 0 (Negative / Ham)**: Legitimate personal or transactional message.
- **Class 1 (Positive / Spam)**: Unsolicited promotional, deceptive, or scam message.

The system learns statistical patterns from a labeled corpus without relying on black-box or non-explainable deep learning models, making every design decision traceable to core undergraduate machine learning theory.

---

## 2. Project Objectives

1. **Dataset Exploration & Hygiene**: Ingest the benchmark UCI SMS Spam Collection, identify class imbalance, and remove duplicate messages to prevent train-test data leakage.
2. **Text Normalization**: Implement a modular preprocessing pipeline that preserves high-signal scam indicators (currency symbols, URLs, numbers, urgency keywords).
3. **Feature Extraction**: Convert text into numerical representations using **Bag-of-Words (CountVectorizer)** with strict train-set vocabulary fitting.
4. **Supervised Model Comparison**: Benchmark four foundational supervised learning algorithms:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Empirical Ablation Experiments**:
   - Impact of vocabulary size ($500$, $1000$, $3000$, $5000$ terms).
   - Impact of preprocessing stages (Minimal vs Standard vs Stopword filtering).
   - Qualitative Error Analysis on False Positives and False Negatives.
6. **Production Artifact & Interactive Demo**: Save a reusable Scikit-learn Pipeline using `joblib` and deploy a clean Streamlit web application.

---

## 3. Technology Stack

- **Language**: Python 3.9+ (tested on Python 3.13)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`sklearn`)
- **Data Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib
- **Interactive UI**: Streamlit
- **Notebook Environment**: Jupyter Notebook

---

## 4. Dataset Description

The project uses the **UCI SMS Spam Collection** dataset:
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Raw Volume**: 5,574 labeled SMS messages.
- **Label Mapping**: `ham` $\rightarrow 0$, `spam` $\rightarrow 1$.

### Data Cleaning and Leakage Prevention
Raw SMS corpora frequently contain duplicate messages (e.g. repeated mass-broadcast promos or identical short responses like *"Ok"*). If duplicates are split naively, identical samples appear in both training and test sets, artificially inflating test scores.

| Metric | Raw Dataset | Cleaned (Unique) Dataset |
| :--- | :--- | :--- |
| **Total Messages** | 5,574 | **5,160** |
| **Duplicates Removed** | — | **414** |
| **Legitimate (Ham - 0)** | 4,827 (86.60%) | **4,518 (87.56%)** |
| **Spam / Scam (Spam - 1)** | 747 (13.40%) | **642 (12.44%)** |
| **Avg Ham Message Length** | 71.02 chars | **70.86 chars** |
| **Avg Spam Message Length** | 138.86 chars | **137.41 chars** |

**Observation**: Spam messages are on average nearly **twice as long** as legitimate messages and frequently contain alphanumeric prize codes, telephone numbers, and URLs.

---

## 5. Machine Learning Pipeline Architecture

```
Raw SMS Message
       │
       ▼
[ Data Cleaning ] ────────► Remove 414 exact duplicates; Map labels (ham:0, spam:1)
       │
       ▼
[ Stratified Split ] ─────► 80% Train (4,128 msgs) | 20% Test (1,032 msgs) [stratify=y]
       │
       ▼
[ Text Preprocessing ] ───► Lowercase, URL/Currency normalization, punctuation strip
       │
       ▼
[ Bag-of-Words (BoW) ] ──► CountVectorizer (max_features=3000) fitted ONLY on Train
       │
       ▼
[ Model Benchmarking ] ──► MNB vs Logistic Regression vs Decision Tree vs Random Forest
       │
       ▼
[ Model Evaluation ] ────► Accuracy, Precision, Recall, F1, Confusion Matrix, 5-Fold CV
       │
       ▼
[ Pipeline Serialization]► models/best_model.pkl (Scikit-Learn Pipeline via Joblib)
       │
       ▼
[ Streamlit App ] ────────► Real-time interactive inference with probability estimates
```

---

## 6. Theory & Supervised Algorithms

### 6.1 Feature Extraction: Bag-of-Words (BoW)
The Bag-of-Words model represents text as a multiset of words, disregarding syntactic word order but recording term occurrence frequency:
$$\mathbf{x} = [c(w_1), c(w_2), \dots, c(w_V)] \in \mathbb{R}^V$$
where $V$ is the vocabulary size (set to $3,000$) and $c(w_j)$ is the count of word $j$ in the message.
- Stored as a **SciPy CSR sparse matrix** ($>99.5\%$ sparse) to conserve memory.
- **Strict Separation Rule**: The vocabulary is learned solely from training data via `fit_transform()`. Test data is only projected via `transform()`.

### 6.2 Classification Algorithms

1. **Multinomial Naive Bayes (`MultinomialNB`)**:
   - Probabilistic classifier assuming conditional feature independence given class $c$:
     $$P(c|\mathbf{x}) \propto P(c) \prod_{i=1}^{V} P(w_i|c)^{x_i}$$
   - Uses Laplace smoothing ($\alpha = 1.0$) to avoid zero probabilities for unseen words.
   - Extremely fast and resilient in high-dimensional discrete count spaces.

2. **Logistic Regression (`LogisticRegression`)**:
   - Linear classifier computing posterior probability via the logistic sigmoid function:
     $$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   - Trained with L2 regularization (`max_iter=1000`, `random_state=42`).
   - Assigns interpretable positive weights to spam words and negative weights to ham words.

3. **Decision Tree Classifier (`DecisionTreeClassifier`)**:
   - Non-linear recursive partitioning based on Gini impurity:
     $$Gini = 1 - \sum_{k} p_k^2$$
   - Regularized with `max_depth=30` to prevent deep branch memorization (overfitting).

4. **Random Forest Classifier (`RandomForestClassifier`)**:
   - Ensemble bagging method constructing 100 decorrelated decision trees (`n_estimators=100`, `max_depth=30`).
   - Aggregates individual tree votes to reduce model variance.

---

## 7. Actual Experimental Results

All experiments were executed on an unseen stratified test partition of **1,032 messages** (904 Ham, 128 Spam).

### Experiment 1: Algorithm Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | False Positives | False Negatives | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.55%** | **94.49%** | **93.75%** | **0.9412** | 7 | **8** | **0.9435** |
| **Logistic Regression** | 97.97% | 98.20% | 85.16% | 0.9121 | **2** | 19 | 0.9167 |
| **Decision Tree** | 97.77% | 94.87% | 86.72% | 0.9061 | 6 | 17 | 0.8822 |
| **Random Forest** | 96.90% | **98.98%** | 75.78% | 0.8584 | **1** | 31 | 0.8680 |

#### Analysis & Best Model Selection:
- **Multinomial Naive Bayes** achieved the highest overall **F1-Score (0.9412)** and highest **Recall (93.75%)**, successfully identifying 120 out of 128 spam messages.
- **Random Forest** and **Logistic Regression** achieved extreme precision ($98.98\%$ and $98.20\%$), yielding almost zero false alarms, but missed 31 and 19 spam messages respectively (lower recall).
- Because a spam filter requires a balanced trade-off between catching deceptive scams (recall) and avoiding misclassifying important personal communications (precision), **Multinomial Naive Bayes** was selected as the production model.

---

### Experiment 2: Vocabulary Size Impact

Fixed classifier: *Multinomial Naive Bayes*

| Vocabulary Size | Test Accuracy | Precision | Recall | Test F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **500 words** | 97.38% | 87.97% | 91.41% | 0.8966 |
| **1000 words** | 98.16% | 92.25% | 92.97% | 0.9261 |
| **3000 words** | **98.55%** | **94.49%** | **93.75%** | **0.9412** |
| **5000 words** | **98.55%** | **94.49%** | **93.75%** | **0.9412** |

**Finding**: Expanding vocabulary from 500 to 3,000 terms yields a substantial +4.46% improvement in F1-score as rare but predictive scam words are incorporated. Performance plateaus at 3,000 terms because the training corpus contains ~3,700 unique words after cleaning; expanding beyond 3,000 introduces diminishing returns and redundant matrix columns.

---

### Experiment 3: Preprocessing Strategies

Fixed classifier: *Multinomial Naive Bayes* ($V = 3000$)

| Strategy | Description | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A: Minimal (Raw)** | Raw text stripped, standard whitespace | 98.26% | 94.35% | 91.41% | 0.9286 |
| **B: Standard** | Lowercase, punctuation strip, URL/Currency tokens | 98.55% | 94.49% | 93.75% | 0.9412 |
| **C: With Stopwords** | Standard + English stopword removal | **98.74%** | **94.57%** | **95.31%** | **0.9494** |

**Finding**: Normalizing currency symbols and URLs (Strategy B) boosted recall from 91.41% to 93.75%. Removing common non-informative English stop words (Strategy C) further reduced background noise, lifting F1-score to 0.9494.

---

### Experiment 4: Misclassification & Error Analysis

The best model made **15 total errors** on the 1,032 test samples:
- **7 False Positives (Ham incorrectly classified as Spam)**
- **8 False Negatives (Spam missed by the model)**

#### 1. False Positive Samples (Legitimate flagged as Spam):
- *"I (Career Tel) have added u as a contact on INDYAROCKS.COM to send FREE SMS. To remove from phonebook - sms NO to &lt;#&gt;"*
  - **Reason**: Contains promotional terminology (`"FREE SMS"`, `"INDYAROCKS.COM"`, `"sms NO"`) characteristic of third-party invite systems that structurally mimic mass marketing.
- *"Have you laid your airtel line to rest?"*
  - **Reason**: Telecom provider brand names (`"airtel"`) heavily co-occur with network marketing spam in the training vocabulary.
- *"Nokia phone is lovly.."*
  - **Reason**: The brand token `"nokia"` frequently appeared in 2000s lottery prize scams (*"Win a new Nokia 6600!"*).

#### 2. False Negative Samples (Spam missed by the model):
- *"Sorry I missed your call let's talk when you have the time. I'm on 07090201529"*
  - **Reason**: Highly sophisticated smishing: intentionally drafted in casual, polite conversational register (`"Sorry I missed your call"`) to disguise a premium-rate callback scam number.
- Non-standard spellings or obfuscated adult chat terms (e.g. `"filthyguys"`, `"150ppmsg"`, `"XXX SLO"`) that fell outside the top vocabulary frequencies.

---

## 8. Project Directory Structure

```
spam-scam-classifier/
│
├── dataset/
│   ├── SMSSpamCollection          # UCI dataset raw text file
│   ├── spam.csv                   # Clean CSV copy
│   └── readme                     # Official UCI dataset documentation
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis notebook
│   ├── 02_preprocessing.ipynb     # Preprocessing & BoW vectorization notebook
│   ├── 03_model_training.ipynb    # Model training & cross-validation notebook
│   └── 04_experiments.ipynb       # Ablation studies & error analysis notebook
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data ingestion, cleaning & duplicate removal
│   ├── preprocessing.py           # Text normalization & stopword filtering
│   ├── feature_extraction.py      # Stratified split & CountVectorizer pipeline
│   ├── train.py                   # 4-model training, 5-fold CV & saving pipeline
│   ├── evaluate.py                # Performance metrics & confusion matrix heatmaps
│   ├── experiments.py             # Ablation studies (Vocab size, Preprocessing, Errors)
│   ├── eda.py                     # Visualizations generator for reports
│   └── predict.py                 # Real-time inference module for Streamlit
│
├── models/
│   ├── best_model.pkl             # Serialized end-to-end Pipeline (BoW + MultinomialNB)
│   ├── count_vectorizer.pkl       # Serialized Bag-of-Words vectorizer
│   ├── model_multinomial_nb.pkl   # Checkpoint: MultinomialNB
│   ├── model_logistic_regression.pkl # Checkpoint: Logistic Regression
│   ├── model_decision_tree.pkl    # Checkpoint: Decision Tree
│   └── model_random_forest.pkl    # Checkpoint: Random Forest
│
├── reports/
│   ├── figures/                   # 8 high-resolution PNG charts
│   │   ├── 01_class_counts.png
│   │   ├── 02_class_percentage_distribution.png
│   │   ├── 03_message_length_distribution.png
│   │   ├── 04_top_words_spam.png
│   │   ├── 05_top_words_ham.png
│   │   ├── 06_model_comparison.png
│   │   ├── 07_vocab_size_comparison.png
│   │   ├── 08_preprocessing_comparison.png
│   │   ├── cm_multinomial_naive_bayes.png
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_decision_tree.png
│   │   └── cm_random_forest.png
│   └── results/                   # Experimental CSV summaries
│       ├── model_comparison.csv
│       ├── experiment_vocab_size.csv
│       ├── experiment_preprocessing.csv
│       └── misclassifications.csv
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Locked dependencies
├── README.md                      # Comprehensive project documentation
└── .gitignore                     # Git ignore rules
```

---

## 9. Installation & Reproduction Guide

### Step 1: Clone or Navigate to the Repository
```bash
cd SpamClassifier
```

### Step 2: Set Up a Python Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Pipeline
Execute the modules sequentially to reproduce all models, figures, and results:

```bash
# 1. Run Data Loader & Exploratory Data Analysis
python src/eda.py

# 2. Train All 4 Models & Evaluate on Unseen Test Data
python src/train.py

# 3. Run Ablation Studies (Vocab size, Preprocessing, Error Analysis)
python src/experiments.py

# 4. Test Single-Sentence Inference
python src/predict.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom SMS messages interactively.

---

## 10. Educational Discussion & Limitations

### Why Bag-of-Words and Classical Models?
1. **Explainability**: Unlike transformer-based models, every decision made by Naive Bayes or Logistic Regression can be traced directly to individual word frequencies and likelihood ratios.
2. **Computational Efficiency**: The entire training pipeline finishes in under 5 seconds on standard CPU hardware without requiring GPUs.
3. **Sparse Linear Bounds**: Text naturally maps to high-dimensional, sparse feature spaces where linear boundaries (Logistic Regression) and conditional probabilities (Naive Bayes) perform well.

### Limitations
- **Word Order Obliviousness**: Bag-of-Words cannot distinguish *"I won nothing from the lottery"* from *"You won the lottery"*. Both contain identical word tokens.
- **Out-of-Vocabulary (OOV) Smishing**: Adversaries who deliberately use leetspeak, phonetic substitutions, or novel vocabulary (e.g. newly minted crypto scams) may bypass a static vocabulary.
- **Operational Scope**: The model detects statistical text patterns. It does not inspect IP headers, telecom metadata, or simulate browser landing pages.
