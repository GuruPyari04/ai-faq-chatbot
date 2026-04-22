import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Data
faq_data = [
    {"question":"What is refund policy?", "answer":"Refund is allowed within 7 days."},
    {"question":"How to contact support?", "answer":"Email us at support@company.com"},
    {"question":"What is delivery time?", "answer":"Delivery takes 3-5 business days."},
    {"question":"What is salary process?", "answer":"Salary is credited on last working day."}
]

questions = [item["question"] for item in faq_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def ask_question(user_query):
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X)
    index = similarity.argmax()
    return faq_data[index]["answer"]

# UI
st.title("🤖 AI FAQ Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    answer = ask_question(user_input)
    st.write("💬 Bot:", answer)