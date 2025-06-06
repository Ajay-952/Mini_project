import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st
import joblib

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    return data

# Train Model
@st.cache_resource
def train_model(data):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data["message"], data["label"])
    return model

# Streamlit UI
def main():
    st.title("📧 Email Spam Detection")
    st.write("Enter an email message below to check if it's spam or not.")

    data = load_data()
    model = train_model(data)

    user_input = st.text_area("✉️ Write your email content here:")

    if st.button("Check"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            prediction = model.predict([user_input])[0]
            if prediction == "spam":
                st.error("⚠️ This email is likely **SPAM**.")
            else:
                st.success("✅ This email is likely **NOT SPAM**.")

if __name__ == "__main__":
    main()
