import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split()  
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("📩 Email Spam Classifier")

input_sms = st.text_area("Enter the email content")

if st.button('Predict'):
    
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]


    if result == 1:
        st.error("❌ This is a Spam email!")
    else:
        st.success("✅ This is a Ham email (not spam).")
