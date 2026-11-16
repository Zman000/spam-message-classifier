import streamlit as st
from src.predict import load_model, predict_message

st.set_page_config(page_title="SMS Spam Detector", layout="centered")

st.title("SMS Spam & Scam Detector")
st.write("Enter an SMS text below to check whether it is legitimate or spam.")

# Sample messages to quickly test
samples = {
    "Select a sample...": "",
    "Lottery Prize (Spam)": "Congratulations! You won a £1,000 Walmart gift card. Click http://claim.com now to get your code!",
    "Bank Alert (Spam)": "URGENT: Your bank account has been locked. Call 0800-123456 immediately to verify.",
    "Casual Chat (Ham)": "Hey, are you free to grab lunch after the lecture today?",
    "Study Notes (Ham)": "Can you please email me the assignment questions from yesterday's class?"
}

selected_sample = st.selectbox("Quick test with a sample message:", list(samples.keys()))
default_text = samples[selected_sample]

user_input = st.text_area("SMS Message Text:", value=default_text, height=120)

if st.button("Classify Message", type="primary"):
    if not user_input.strip():
        st.warning("Please type or select a message first.")
    else:
        model = load_model()
        result = predict_message(user_input, model=model)

        if result["is_spam"]:
            st.error(f"**Prediction: Potential Spam / Scam** (Confidence: {result['confidence']}%)")
        else:
            st.success(f"**Prediction: Legitimate Message (Ham)** (Confidence: {result['confidence']}%)")

        st.write(f"Spam probability: `{result['spam_probability']:.4f}`")

st.markdown("---")
st.caption("Machine Learning project using Bag-of-Words and Multinomial Naive Bayes trained on the UCI SMS Spam Collection.")
