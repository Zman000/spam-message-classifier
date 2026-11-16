# SMS Spam and Scam Message Classifier

A supervised machine learning project to classify SMS messages as legitimate (**Ham**) or suspicious/fraudulent (**Spam**).

Built as an undergraduate machine learning course project demonstrating text preprocessing, Bag-of-Words feature extraction, model comparison, and interactive deployment.

---

## 1. Project Overview

- **Dataset**: UCI SMS Spam Collection (5,160 unique messages after removing duplicates)
- **Labels**: `ham` (0) and `spam` (1)
- **Feature Extraction**: Bag-of-Words using Scikit-Learn `CountVectorizer` (vocabulary size: 3000)
- **Algorithms Compared**:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- **Best Model**: Multinomial Naive Bayes (F1-score: **0.9375**, Accuracy: **98.45%**)

---

## 2. Model Performance Comparison

Evaluated on an unseen 20% test set (1,032 messages):

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **98.45%** | **93.75%** | **93.75%** | **0.9375** |
| **Logistic Regression** | 97.87% | 95.69% | 86.72% | 0.9098 |
| **Decision Tree** | 97.87% | 94.92% | 87.50% | 0.9106 |
| **Random Forest** | 97.09% | 99.00% | 77.34% | 0.8684 |

**Key Takeaways**:
- **Multinomial Naive Bayes** achieved the best overall balance between precision and recall, catching 120 out of 128 spam messages in the test set.
- **Random Forest** had very high precision (99%) but missed 29 spam messages (lower recall: 77.3%).
- Multinomial Naive Bayes was selected as the final model and saved to `models/best_model.pkl`.

---

## 3. Project Structure

```
SpamClassifier/
├── dataset/
│   ├── spam.csv                 # Cleaned dataset (label, message)
│   └── SMSSpamCollection        # UCI raw dataset
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Cleaning & Bag-of-Words
│   ├── 03_model_training.ipynb  # Training & evaluation
│   └── 04_experiments.ipynb     # Ablation studies
├── src/
│   ├── data_loader.py           # Loads and cleans dataset
│   ├── preprocessing.py         # Text cleaning and token normalization
│   ├── feature_extraction.py    # Stratified split & CountVectorizer
│   ├── train.py                 # Trains models & saves best pipeline
│   ├── evaluate.py              # Metric calculations & confusion matrix
│   ├── experiments.py           # Vocabulary and preprocessing experiments
│   ├── eda.py                   # Generates EDA visualizations
│   └── predict.py               # Inference helper for Streamlit
├── models/
│   └── best_model.pkl           # Saved Scikit-Learn pipeline
├── reports/
│   ├── figures/                 # EDA plots & confusion matrices
│   └── results/                 # Metrics CSVs
├── app.py                       # Simple Streamlit web interface
├── requirements.txt             # Dependencies
└── README.md
```

---

## 4. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run EDA & Generate Plots
```bash
python src/eda.py
```

### Train Models & Save Best Pipeline
```bash
python src/train.py
```

### Run Experiments (Vocab Size, Preprocessing, Error Analysis)
```bash
python src/experiments.py
```

### Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test custom SMS messages.
