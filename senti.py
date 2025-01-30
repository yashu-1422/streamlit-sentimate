from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import gdown
import os
import streamlit as st

# Google Drive link
file_id = '1DYHp9NFq0sAFOiKE5Xm5BaHoAlS2Qfp8'
download_url = f'https://drive.google.com/uc?id={file_id}'

# Function to download the dataset if not already downloaded
def download_dataset(url, output_path='IMDB_Dataset.csv'):
    if not os.path.exists(output_path):
        st.write("Downloading dataset...")
        gdown.download(url, output_path, quiet=False)
    else:
        st.write(f"Dataset already exists at {output_path}. Skipping download.")
    return output_path

# Download the dataset
dataset_path = download_dataset(download_url)

# Load the dataset
df = pd.read_csv(dataset_path)

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

df['review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Save the Random Forest model with compression (less than 25MB)
joblib.dump(rf_model, 'Random_Forest_Model.pkl', compress=3)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl', compress=3)

# Evaluate the model
y_pred = rf_model.predict(X_test_tfidf)
st.write("Model: Random Forest")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(classification_report(y_test, y_pred))

# Streamlit Interface
st.title('Sentiment Analysis of Movie Reviews')

st.write("""
This application predicts the sentiment of a movie review (positive/negative) using a Random Forest Classifier model.
You can enter a movie review below, and the system will predict whether it's positive or negative.
""")

# Input from the user
user_input = st.text_area("Enter your movie review:")

# Function to make prediction
def predict_sentiment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review])
    prediction = rf_model.predict(review_tfidf)
    return "Positive" if prediction == 1 else "Negative"

if st.button('Predict Sentiment'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a movie review to get the prediction.")

