"""
app.py
======
Streamlit Interactive Application: SMS Spam & Scam Detector.

Demonstrates real-time inference using the trained Scikit-learn Pipeline
(Bag-of-Words + Multinomial Naive Bayes) developed for the Machine Learning course.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.predict import load_model, predict_message

# Set Page Config
st.set_page_config(
    page_title="SMS Spam & Scam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .result-box-ham {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-box-spam {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .limitation-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_pipeline():
    """Loads and caches the model pipeline to avoid retraining on every interaction."""
    model_path = os.path.join("models", "best_model.pkl")
    return load_model(model_path)


# Sample messages for user exploration
SAMPLE_MESSAGES = {
    "-- Select a sample message to test --": "",
    "Lottery Prize Scam": "Congratulations! You have won a £1,000 Walmart Gift Card. Call 0800-999-123 or click http://claim-rewards.com to claim your prize now! Valid today only.",
    "Banking Phishing Scam": "URGENT: Your bank account has been flagged for suspicious activity. Verify your identity immediately at http://secure-bank-update.com to prevent suspension.",
    "Casual Social Text (Ham)": "Hey! Are you still up for grabbing coffee after our machine learning lecture this afternoon? Let me know.",
    "Academic Coordination (Ham)": "Hi Sarah, can you please email me the assignment questions for chapter 4? I couldn't attend the tutorial yesterday.",
    "Short Informal Chat (Ham)": "Ok cool, I will reach your place in 10 minutes. See you soon!"
}


def main():
    st.markdown('<div class="main-title">🛡️ SMS Spam & Scam Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Supervised Machine Learning Course Project | Bag-of-Words & Multinomial Naive Bayes</div>', unsafe_allow_html=True)

    # Sidebar: Model Specs & Inspection
    st.sidebar.header("📊 Model Specifications")
    st.sidebar.markdown("""
    - **Dataset**: UCI SMS Spam Collection
    - **Clean Samples**: 5,160 unique messages
    - **Algorithm**: Multinomial Naive Bayes
    - **Feature Extraction**: Bag-of-Words (CountVectorizer)
    - **Vocabulary Size**: 3,000 terms
    - **Test Accuracy**: **98.55%**
    - **Test F1-Score**: **0.9412**
    - **Test Precision**: **94.49%**
    - **Test Recall**: **93.75%**
    """)

    # Option to view comparison table
    if st.sidebar.checkbox("Show Model Comparison Table"):
        comp_path = os.path.join("reports", "results", "model_comparison.csv")
        if os.path.exists(comp_path):
            st.sidebar.dataframe(pd.read_csv(comp_path)[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]])

    # Option to view confusion matrix
    if st.sidebar.checkbox("Show Confusion Matrix Heatmap"):
        cm_path = os.path.join("reports", "figures", "cm_multinomial_naive_bayes.png")
        if os.path.exists(cm_path):
            st.sidebar.image(cm_path, caption="Confusion Matrix on Unseen Test Set (1,032 msgs)")

    # Load Model Pipeline
    try:
        pipeline = get_cached_pipeline()
    except Exception as e:
        st.error(f"Error loading trained pipeline: {e}. Please ensure 'models/best_model.pkl' exists.")
        return

    # Sample selector
    selected_sample_title = st.selectbox(
        "Choose a pre-filled sample message, or type your own below:",
        options=list(SAMPLE_MESSAGES.keys()),
        index=0
    )

    initial_text = SAMPLE_MESSAGES[selected_sample_title]

    # Input text area
    user_input = st.text_area(
        "Enter SMS Message Content:",
        value=initial_text,
        height=130,
        placeholder="Type or paste an SMS message here to classify..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_button = st.button("🔍 Classify Message", type="primary", use_container_width=True)

    if classify_button or (initial_text and user_input == initial_text):
        if not user_input.strip():
            st.warning("Please enter or select an SMS message first.")
        else:
            with st.spinner("Analyzing message text..."):
                res = predict_message(user_input, model=pipeline)

            is_spam = res["prediction_num"] == 1

            if is_spam:
                st.markdown(f"""
                <div class="result-box-spam">
                    <h3 style="color: #b91c1c; margin: 0 0 0.5rem 0;">⚠️ Prediction: Potential Spam / Scam</h3>
                    <p style="margin: 0; font-size: 1.05rem;">
                        This message exhibits strong linguistic characteristics typically associated with spam, promotional campaigns, or scam attempts.
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-weight: 600;">
                        Confidence: {res['confidence_percent']} (Spam Probability: {res['spam_probability']:.4f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-ham">
                    <h3 style="color: #047857; margin: 0 0 0.5rem 0;">✅ Prediction: Legitimate Message (Ham)</h3>
                    <p style="margin: 0; font-size: 1.05rem;">
                        This message appears normal and conversational. No strong spam patterns were detected.
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-weight: 600;">
                        Confidence: {res['confidence_percent']} (Spam Probability: {res['spam_probability']:.4f})
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics Breakdown
            st.write("")
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Legitimate Probability", f"{(1.0 - res['spam_probability']):.2%}")
            with col_prob2:
                st.metric("Spam / Scam Probability", f"{res['spam_probability']:.2%}")

            # Detected Vocabulary Tokens
            if res["detected_vocab_words"]:
                st.markdown("**Matched Vocabulary Words in Training Dictionary:**")
                st.write(", ".join([f"`{w}`" for w in res["detected_vocab_words"]]))

    # Educational limitation notice
    st.markdown("""
    <div class="limitation-box">
        <strong>⚠️ Academic & Operational Disclaimer:</strong><br>
        This classification system is an educational machine learning demonstration built with a <em>Bag-of-Words (CountVectorizer)</em>
        feature extractor and a <em>Multinomial Naive Bayes</em> classifier trained on the UCI SMS Spam Collection.
        It identifies statistical text patterns and keyword co-occurrences. It does <strong>not</strong> verify real-time phone number ownership,
        telecom headers, or guarantee fraud verification.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
