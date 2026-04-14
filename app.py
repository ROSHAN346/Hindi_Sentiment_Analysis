import streamlit as st
import pandas as pd
import numpy as np
import codecs
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ==========================
# STOPWORDS
# ==========================
stopwords = codecs.open(
    "hindi_stopwords.txt",
    "r",
    encoding='utf-8',
    errors='ignore'
).read().split('\n')


# ==========================
# TOKENIZER
# ==========================
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    return [w for w in text.split() if w not in stopwords]


# ==========================
# LOAD DATA
# ==========================
def load_data():
    pos = codecs.open("pos_hindi.txt", "r", encoding='utf-8').read()
    neg = codecs.open("neg_hindi.txt", "r", encoding='utf-8').read()

    docs, labels = [], []

    for line in pos.split('$'):
        if line.strip():
            docs.append(line.strip())
            labels.append(1)

    for line in neg.split('$'):
        if line.strip():
            docs.append(line.strip())
            labels.append(0)

    return docs, labels


docs, y = load_data()


# ==========================
# TF-IDF
# ==========================
vectorizer = TfidfVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(docs)


# ==========================
# MODELS
# ==========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SGD": SGDClassifier(max_iter=2000),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB()
}


# Train models
trained_models = {}

for name, model in models.items():
    model.fit(X, y)
    trained_models[name] = model


# ==========================
# UI
# ==========================
st.title("Hindi Sentiment Analysis System 🇮🇳")

text = st.text_area("Enter Hindi Text")

if st.button("Predict"):

    input_vec = vectorizer.transform([text])

    results = []

    for name, model in trained_models.items():

        pred = model.predict(input_vec)[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😡"

        results.append([name, sentiment])

    df = pd.DataFrame(results, columns=["Algorithm", "Prediction"])

    st.table(df)