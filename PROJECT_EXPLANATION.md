# Comprehensive Project Documentation & Code Walkthrough

**Project Title**: SMS Spam and Scam Detection Using Supervised Machine Learning Algorithms  
**Coursework Domain**: Supervised Machine Learning & Natural Language Processing (NLP)  
**Target Audience**: TA Evaluation, Viva Defense, and Academic Review  

---

## Table of Contents
1. [Project Overview & Objective](#1-project-overview--objective)
2. [Dataset: Source, Structure, and Cleaning](#2-dataset-source-structure-and-cleaning)
3. [End-to-End Machine Learning Workflow](#3-end-to-end-machine-learning-workflow)
4. [Libraries & Technologies Used](#4-libraries--technologies-used)
5. [File-by-File Code Breakdown](#5-file-by-file-code-breakdown)
   - [`src/data_loader.py`](#srcdata_loaderpy)
   - [`src/preprocessing.py`](#srcpreprocessingpy)
   - [`src/feature_extraction.py`](#srcfeature_extractionpy)
   - [`src/train.py`](#srctrainpy)
   - [`src/evaluate.py`](#srcevaluatepy)
   - [`src/experiments.py`](#srcexperimentspy)
   - [`src/eda.py`](#srcedapy)
   - [`src/predict.py`](#srcpredictpy)
   - [`app.py`](#apppy)
   - [Notebooks (`notebooks/`)](#notebooks-notebooks)
   - [Model Artifacts (`models/`)](#model-artifacts-models)
   - [Reports & Visualizations (`reports/`)](#reports--visualizations-reports)
6. [Machine Learning Algorithms Explained](#6-machine-learning-algorithms-explained)
7. [Experimental Results & Evaluation Metrics](#7-experimental-results--evaluation-metrics)
8. [Common TA Defense Questions & Answers](#8-common-ta-defense-questions--answers)

---

## 1. Project Overview & Objective

### The Problem
Short Message Service (SMS) remains one of the most widely used channels for personal messaging, bank transaction confirmations, and multi-factor authentication (OTPs). Because traditional cellular SMS lacks built-in sender identity verification, malicious actors exploit SMS for:
- **Smishing (SMS Phishing)**: Impersonating banks or authorities to steal credentials.
- **Financial Scams**: Fake lotteries, prize notifications, and gift card offers.
- **Spam**: Unsolicited commercial broadcasts.

### The Objective
To build an explainable, end-to-end binary classification system that takes raw SMS text as input and predicts:
- **Class 0 (Ham)**: Legitimate message.
- **Class 1 (Spam)**: Fraudulent, deceptive, or spam message.

### Academic Constraints
- **Explainability**: Use foundational supervised machine learning algorithms covered in coursework (Naive Bayes, Logistic Regression, Decision Trees, Random Forests).
- **Feature Extraction**: Use Bag-of-Words (`CountVectorizer`) rather than black-box deep learning or complex transformer embeddings.
- **Rigorous ML Hygiene**: Prevent train-test data leakage by removing duplicate messages prior to splitting and fitting feature extractors only on training data.

---

## 2. Dataset: Source, Structure, and Cleaning

### Source
The project uses the benchmark **UCI SMS Spam Collection Dataset** collected by Tiago A. Almeida and José María Gómez Hidalgo from the UCI Machine Learning Repository.
- **Dataset URL**: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

### Raw Structure
The raw file (`dataset/SMSSpamCollection`) is a tab-separated text file without headers:
- Column 1: Label (`ham` or `spam`)
- Column 2: Raw message text

### Data Cleaning & Duplicate Handling
Raw SMS datasets naturally contain identical messages repeated multiple times (such as repeated spam broadcasts or short replies like *"Ok"*, *"Call you later"*).

| Metric | Raw Dataset | Cleaned Dataset |
| :--- | :--- | :--- |
| **Total Messages** | 5,574 | **5,160** |
| **Duplicates Removed** | — | **414** |
| **Legitimate (Ham / Class 0)** | 4,827 (86.6%) | **4,518 (87.6%)** |
| **Spam / Scam (Spam / Class 1)** | 747 (13.4%) | **642 (12.4%)** |
| **Average Ham Length** | 71.0 characters | **70.9 characters** |
| **Average Spam Length** | 138.9 characters | **137.4 characters** |

### Why Duplicate Removal is Critical (Data Leakage Prevention)
If duplicates are not removed before splitting the data, the exact same promotional or conversational messages will end up in both the training set and the test set. The model would merely memorize those samples rather than generalizing to unseen messages, resulting in artificially inflated evaluation scores.

---

## 3. End-to-End Machine Learning Workflow

```
1. Raw Data Ingestion (dataset/spam.csv)
       │
       ▼
2. Data Cleaning & Label Encoding (ham -> 0, spam -> 1; drop 414 duplicates)
       │
       ▼
3. Exploratory Data Analysis (Class balance, length distribution, top tokens)
       │
       ▼
4. Stratified Train-Test Split (80% Train = 4,128 msgs, 20% Test = 1,032 msgs)
       │
       ▼
5. Text Preprocessing (Lowercase, normalize URLs & Currency, strip punctuation)
       │
       ▼
6. Bag-of-Words Feature Extraction (CountVectorizer, max_features=3000)
       │  * Fitted strictly on training set only (fit_transform)
       ▼  * Test set strictly transformed (transform)
7. Supervised Model Training (MultinomialNB, LogisticRegression, DecisionTree, RandomForest)
       │
       ▼
8. Evaluation & Comparison (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
       │
       ▼
9. Ablation Studies (Vocab sizes, Preprocessing variations, Error analysis)
       │
       ▼
10. Model Serialization (Pipeline saved to models/best_model.pkl via Joblib)
       │
       ▼
11. Deployment via Streamlit Web App (app.py)
```

---

## 4. Libraries & Technologies Used

1. **Python 3.9+** (Tested on Python 3.13): Core programming language.
2. **Pandas**: Tabular data manipulation, CSV ingestion, label mapping, and metric tabulation.
3. **NumPy**: Numerical operations and probability score transformations.
4. **Scikit-Learn (`sklearn`)**:
   - `model_selection.train_test_split`: Stratified 80:20 data splitting.
   - `feature_extraction.text.CountVectorizer`: Bag-of-Words vocabulary construction and word count matrix generation.
   - `naive_bayes.MultinomialNB`: Multinomial Naive Bayes classifier.
   - `linear_model.LogisticRegression`: Logistic Regression classifier.
   - `tree.DecisionTreeClassifier`: Decision tree classifier.
   - `ensemble.RandomForestClassifier`: Random forest ensemble classifier.
   - `metrics`: Evaluation functions (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).
   - `pipeline.Pipeline`: Encapsulating vectorizer and model into a single deployable object.
5. **Matplotlib & Seaborn**: Generating academic visualizations (count plots, histograms, bar charts, confusion matrix heatmaps).
6. **Joblib**: Efficient serialization (pickling) of trained model pipelines to disk.
7. **Streamlit**: Lightweight Python-native web application framework for live interactive demonstrations.
8. **Jupyter Notebook (`.ipynb`)**: Interactive documents for step-by-step code execution and visual verification.

---

## 5. File-by-File Code Breakdown

```
SpamClassifier/
├── dataset/
├── notebooks/
├── src/
├── models/
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

### `src/data_loader.py`
- **Purpose**: Ingests the raw SMS data, cleans the dataframe, and maps binary labels.
- **Key Functions**:
  - `load_data(filepath)`: Reads either `dataset/spam.csv` or `dataset/SMSSpamCollection`. Standardizes columns to `label` and `message`. Strips whitespace, drops missing values, filters for valid `ham`/`spam` values, drops duplicate messages, and maps `ham -> 0` and `spam -> 1`.
  - `clean_dataset(df)`: Helper function returning the clean dataframe and summary counts.
- **Design Decision**: Returns `label_num` as integer target ($0$ or $1$) suitable for Scikit-Learn estimators.

### `src/preprocessing.py`
- **Purpose**: Normalizes raw SMS text into clean tokens while explicitly preserving spam indicators.
- **Key Functions**:
  - `clean_text(text)`: 
    1. Converts text to lowercase.
    2. Replaces URLs (`http://...`, `www....`) with token `urladdr`.
    3. Replaces currency symbols (`$`, `£`, `€`) with token `moneyprize`.
    4. Replaces punctuation and special characters with spaces (`re.sub(r"[^\w\s]", " ", text)`).
    5. Normalizes multiple spaces into a single space.
  - `preprocess_text(text, mode)`: Supports three experimentation modes:
    - `"minimal"`: Keeps original raw text.
    - `"standard"`: Lowercasing, URL/currency placeholder, and punctuation cleaning.
    - `"with_stopwords"`: Standard cleaning plus removal of common English stop words (`STOPWORDS`).
  - `preprocess_corpus(corpus, mode)`: Batch processes a list or pandas Series of messages.
- **Design Decision**: Retains numbers, currency tokens, and URL markers because these features carry high discriminative signal for spam detection.

### `src/feature_extraction.py`
- **Purpose**: Splits data into stratified train/test partitions and extracts Bag-of-Words matrices.
- **Key Functions**:
  - `split_data(df, test_size=0.2, random_state=42)`: Uses `train_test_split` with `stratify=df['label_num']` to guarantee both splits retain the exact same 87.6% : 12.4% class ratio.
  - `extract_bow_features(X_train, X_test, max_features=3000, preprocessing_mode='standard')`:
    - Preprocesses `X_train` and `X_test`.
    - Initializes `CountVectorizer(max_features=3000)`.
    - Calls `.fit_transform()` on `X_train` (learns vocabulary strictly on training data).
    - Calls `.transform()` on `X_test` (projects test data into learned vocabulary).
- **Design Decision**: Stores matrices in SciPy CSR sparse matrix format to handle 99.55% zero entries without consuming unnecessary RAM.

### `src/evaluate.py`
- **Purpose**: Computes binary classification metrics and renders visual confusion matrices.
- **Key Functions**:
  - `calculate_metrics(y_true, y_pred, model_name)`: Computes Accuracy, Precision, Recall, and F1-Score treating Spam ($1$) as the positive class (`zero_division=0`).
  - `plot_confusion_matrix(y_true, y_pred, title, save_path)`: Uses Seaborn `heatmap` to plot actual vs. predicted labels (True Negatives, False Positives, False Negatives, True Positives) and saves high-resolution PNGs to `reports/figures/`.

### `src/train.py`
- **Purpose**: Orchestrates model training across four algorithms, evaluates them on test data, and serializes the winning model.
- **Key Functions**:
  - `train_models()`:
    1. Loads dataset via `load_data()`.
    2. Splits into Train (4,128) and Test (1,032).
    3. Vectorizes training and test sets using CountVectorizer.
    4. Trains all 4 models:
       - `MultinomialNB()`
       - `LogisticRegression(max_iter=1000, random_state=42)`
       - `DecisionTreeClassifier(max_depth=30, random_state=42)`
       - `RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)`
    5. Evaluates each on the unseen test set and saves confusion matrix plots.
    6. Saves metric summary table to `reports/results/model_comparison.csv` and comparison bar chart to `reports/figures/06_model_comparison.png`.
    7. Encapsulates the best model (`MultinomialNB`) and `CountVectorizer(preprocessor=clean_text)` into a unified `Pipeline` and saves it to `models/best_model.pkl` using `joblib.dump()`.

### `src/experiments.py`
- **Purpose**: Executes the three required academic ablation experiments.
- **Key Functions**:
  - `run_vocab_size_experiment()`: Evaluates Multinomial Naive Bayes across vocabulary limits of 500, 1000, 3000, and 5000 words. Saves results to `reports/results/experiment_vocab_size.csv` and `reports/figures/07_vocab_size_comparison.png`.
  - `run_preprocessing_experiment()`: Compares Minimal (raw), Standard (cleaned), and Stopwords Removed pipelines. Saves results to `reports/results/experiment_preprocessing.csv` and `reports/figures/08_preprocessing_comparison.png`.
  - `run_misclassification_analysis()`: Identifies all test messages where predictions failed, labels them as False Positives or False Negatives, and exports them to `reports/results/misclassifications.csv`.

### `src/eda.py`
- **Purpose**: Generates exploratory charts and prints statistical distributions.
- **Key Functions**:
  - `run_eda()`:
    - Calculates total messages, class counts, percentages, and mean character lengths.
    - Generates and saves:
      1. `01_class_counts.png`: Bar plot of Ham vs. Spam counts.
      2. `02_class_percentage_distribution.png`: Pie chart showing class imbalance.
      3. `03_message_length_distribution.png`: Histogram comparing message character lengths.
      4. `04_top_words_spam.png`: Top frequent tokens in spam messages.
      5. `05_top_words_ham.png`: Top frequent tokens in legitimate messages.

### `src/predict.py`
- **Purpose**: Provides inference functions for single messages, used by `app.py`.
- **Key Functions**:
  - `load_model(path)`: Loads and caches `models/best_model.pkl`.
  - `predict_message(text, model)`: Calls `.predict()` and `.predict_proba()` on input text. Returns dictionary containing predicted label (`"Ham"` or `"Spam"`), category description, confidence percentage, and spam probability.

### `app.py`
- **Purpose**: Clean, lightweight Streamlit web application.
- **Features**:
  - Title and brief description.
  - Quick-test sample message selector (Lottery scam, Bank alert, Casual text, Study notes).
  - Text area for custom user input.
  - "Classify Message" action button.
  - Output display: `st.success()` for Ham (Legitimate) and `st.error()` for Spam/Scam, with confidence percentages and exact probabilities.
  - Footer disclaimer noting model capabilities and limitations.

### Notebooks (`notebooks/`)
Interactive step-by-step notebooks mirroring each phase:
- `01_eda.ipynb`: Data inspection, duplicate checking, and chart rendering.
- `02_preprocessing.ipynb`: Demonstration of text cleaning and CountVectorizer feature shapes.
- `03_model_training.ipynb`: Execution of the 4 classifiers and model comparison table.
- `04_experiments.ipynb`: Ablation graphs and misclassified sample inspections.

### Model Artifacts (`models/`)
- `best_model.pkl`: Reusable Scikit-Learn Pipeline combining the `CountVectorizer` preprocessor and the fitted `MultinomialNB` classifier.

### Reports (`reports/`)
- `reports/figures/`: High-resolution PNG visual outputs (12 plots total).
- `reports/results/`: Metric comparison CSV tables.

---

## 6. Machine Learning Algorithms Explained

### 1. Multinomial Naive Bayes (`MultinomialNB`)
- **Concept**: A probabilistic classifier based on Bayes' Theorem:
  $$P(y|x) = \frac{P(y) \prod_{i=1}^n P(x_i|y)}{P(x)}$$
- **Assumption**: Features (words) are conditionally independent given the class label. While this assumption is linguistically unrealistic (words correlate), Naive Bayes performs remarkably well in practice for text classification.
- **Laplace Smoothing ($\alpha = 1.0$)**: Adds a pseudo-count of 1 to every word frequency to prevent zero-probability multiplication when an unseen word appears.

### 2. Logistic Regression (`LogisticRegression`)
- **Concept**: A linear model that estimates the probability of binary membership using the logistic sigmoid function:
  $$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
- **Behavior**: Learns positive weights for strong spam words (e.g., `claim`, `prize`, `call`) and negative weights for conversational ham words (e.g., `my`, `ok`, `later`).

### 3. Decision Tree Classifier (`DecisionTreeClassifier`)
- **Concept**: A non-linear rule-based tree that recursively splits samples on word occurrences to maximize purity (minimizing Gini Impurity):
  $$\text{Gini} = 1 - \sum_{k} p_k^2$$
- **Regularization**: `max_depth=30` prevents individual branches from splitting indefinitely, preventing the tree from memorizing the training set.

### 4. Random Forest Classifier (`RandomForestClassifier`)
- **Concept**: An ensemble method combining 100 decorrelated decision trees (`n_estimators=100`).
- **Bagging & Feature Subsampling**: Each tree is trained on a bootstrap sample of the training data and splits on a random subset of features. The majority vote across all 100 trees determines the final prediction, reducing variance compared to a single decision tree.

---

## 7. Experimental Results & Evaluation Metrics

All models were evaluated on the **same unseen test set** of **1,032 messages** (904 Ham, 128 Spam).

### Primary Model Comparison (Experiment 1)

| Model | Test Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.45%** | **93.75%** | **93.75%** | **0.9375** | 8 | **8** |
| **Logistic Regression** | 97.87% | 95.69% | 86.72% | 0.9098 | 5 | 17 |
| **Decision Tree** | 97.87% | 94.92% | 87.50% | 0.9106 | 6 | 16 |
| **Random Forest** | 97.09% | 99.00% | 77.34% | 0.8684 | **1** | 29 |

### Why Multinomial Naive Bayes is the Best Choice
1. **Precision-Recall Balance**: In spam filtering, **Recall** is critical (we do not want malicious smishing or financial scams escaping detection and landing in the user's inbox).
2. **Random Forest vs. Naive Bayes**: Random Forest achieved near-perfect precision (99.00%), meaning it almost never misclassified legitimate messages. However, its recall was only 77.34% (it missed 29 spam messages).
3. **Winning Model**: Multinomial Naive Bayes achieved **93.75% Recall** (catching 120 of 128 spam messages) while maintaining **93.75% Precision**, yielding the highest **F1-Score (0.9375)** and highest **Accuracy (98.45%)**.

---

### Vocabulary Size Ablation (Experiment 2)
Fixed Classifier: *Multinomial Naive Bayes*

| Vocabulary Size | Test Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **500 words** | 97.67% | 90.00% | 91.41% | 0.9070 |
| **1,000 words** | 97.97% | 90.84% | 92.97% | 0.9189 |
| **3,000 words** | **98.45%** | **93.75%** | **93.75%** | **0.9375** |
| **5,000 words** | 98.55% | 95.20% | 92.97% | 0.9407 |

*Finding*: Expanding vocabulary from 500 to 3,000 words provides a sharp gain in F1-score (+3.05%). Beyond 3,000 words, gains plateau because the entire cleaned vocabulary contains ~3,700 unique words.

---

### Preprocessing Variations (Experiment 3)
Fixed Classifier: *Multinomial Naive Bayes* ($V = 3,000$)

| Strategy | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Minimal (Raw text)** | 98.35% | 93.70% | 92.97% | 0.9333 |
| **Standard (Punctuation, URLs, Currency)** | 98.45% | 93.75% | 93.75% | 0.9375 |
| **Stopwords Removed** | **98.55%** | **93.80%** | **94.53%** | **0.9416** |

*Finding*: Cleaning punctuation and mapping currency/URLs improved recall from 92.97% to 93.75%. Stopword removal provided a minor additional boost by filtering uninformative background noise.

---

### Misclassification Analysis (Experiment 4)
The best model produced only **16 errors** across 1,032 test messages:
- **8 False Positives** (Legitimate messages classified as Spam):
  - Example: Messages containing commercial invite words like *"FREE SMS"* or provider brand names like *"airtel"* (co-occur frequently with promos).
- **8 False Negatives** (Spam messages classified as Ham):
  - Example: *"Sorry I missed your call let's talk when you have the time. I'm on 07090201529"*.
  - Reason: The scammer deliberately adopted conversational, polite language (*"Sorry I missed your call let's talk"*) to trick the bag-of-words model into assigning ham probabilities, hiding a premium-rate callback scam.

---

## 8. Common TA Defense Questions & Answers

**Q1: Why did you remove duplicate messages before splitting the data?**  
> *"In real-world SMS collections, marketing blasts and short replies like 'Ok' repeat often. If we split without dropping duplicates, identical messages land in both train and test sets, causing data leakage and giving artificially high test accuracy."*

**Q2: Why did you choose Bag-of-Words instead of TF-IDF or Word Embeddings?**  
> *"The project objective focuses on explainable supervised learning fundamentals. CountVectorizer represents raw integer word frequencies, which directly aligns with the multinomial event model assumption of Multinomial Naive Bayes. Every decision can be explained by word count likelihoods."*

**Q3: Why did Multinomial Naive Bayes outperform Random Forest on F1-Score?**  
> *"Text classification with Bag-of-Words produces high-dimensional, sparse feature spaces (3,000 columns, 99.5% zeros). Probabilistic models like Naive Bayes handle sparse independent word counts effectively. Random Forest achieved high precision (99%), but was too conservative and missed 29 spam messages (low recall: 77.34%). Naive Bayes caught 120 of 128 spam messages (93.75% recall) with 93.75% precision."*

**Q4: How does the trained model run in the Streamlit application?**  
> *"We bundle the `CountVectorizer` preprocessor and the `MultinomialNB` classifier into a Scikit-Learn `Pipeline` and serialize it as `models/best_model.pkl` using Joblib. When `app.py` runs, it loads this pre-trained file in under 0.1 seconds, eliminating the need to re-train the model."*

**Q5: What are the main limitations of this system?**  
> *"Because Bag-of-Words ignores word order, the model cannot distinguish 'I won nothing from the lottery' from 'You won the lottery'. Additionally, it only inspects message text; it does not check network telecom headers or phone number carrier records."*
