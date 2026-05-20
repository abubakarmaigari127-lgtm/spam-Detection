import streamlit as st
import pandas as pd
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
import time

# Page config
st.set_page_config(page_title="Spam Detection System", page_icon="📧", layout="wide")

# Download stopwords
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\d+', '', text)
    return text

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("📧 Spam Detection System")
st.markdown("""
This application uses a **Machine Learning (Naive Bayes)** model to classify text messages as either **Spam** or **Ham (Not Spam)**.
The model was trained on the standard SMS Spam Collection dataset using TF-IDF vectorization.
""")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Message Analysis")
    user_input = st.text_area("Enter the message you want to check:", height=150, placeholder="Type or paste your message here...")
    
    predict_btn = st.button("🔍 Predict")
    
    if predict_btn:
        if user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing message..."):
                # Preprocess and predict
                cleaned_text = preprocess_text(user_input)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)[0]
                probability = model.predict_proba(vectorized_text)[0]
                
                label = "Spam" if prediction == 1 else "Not Spam (Ham)"
                confidence = probability[prediction] * 100
                
                # Result display
                st.markdown("---")
                if prediction == 1:
                    st.error(f"### Prediction: {label}")
                else:
                    st.success(f"### Prediction: {label}")
                
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Add to history
                st.session_state.history.append({
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Message": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "Prediction": label,
                    "Confidence": f"{confidence:.2f}%"
                })

with col2:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for batch processing", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if 'message' in batch_df.columns:
                with st.spinner("Processing batch..."):
                    batch_df['Cleaned'] = batch_df['message'].apply(preprocess_text)
                    batch_vec = vectorizer.transform(batch_df['Cleaned'])
                    batch_df['Prediction'] = model.predict(batch_vec)
                    batch_df['Prediction'] = batch_df['Prediction'].map({0: 'Ham', 1: 'Spam'})
                    
                    st.write("Batch Results Preview:")
                    st.dataframe(batch_df[['message', 'Prediction']].head())
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error("CSV must contain a 'message' column.")
        except Exception as e:
            st.error(f"Error: {e}")

# History Section
st.markdown("---")
st.subheader("📜 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No prediction history yet.")

# Model Metrics
with st.expander("📊 View Model Performance Metrics"):
    st.image("confusion_matrix.png", caption="Confusion Matrix from Training")
    st.write("The model achieved **96.77% accuracy** on the test set.")
    st.markdown("""
    - **Precision (Ham):** 0.96
    - **Recall (Ham):** 1.00
    - **Precision (Spam):** 1.00
    - **Recall (Spam):** 0.76
    """)
