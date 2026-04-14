import streamlit as st
import pandas as pd
import numpy as np
import codecs
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
# TF FEATURES
# ==========================
tf_vectorizer = CountVectorizer(tokenizer=tokenize)
X_tf = tf_vectorizer.fit_transform(docs)


# ==========================
# TF-IDF FEATURES
# ==========================
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    ngram_range=(1,2),
    max_features=2000
)

X_tfidf = tfidf_vectorizer.fit_transform(docs)


# ==========================
# MODELS
# ==========================
models = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "SGD": lambda: SGDClassifier(max_iter=2000),
    "SVM": lambda: SVC(kernel='linear'),
    "KNN": lambda: KNeighborsClassifier(),
    "Decision Tree": lambda: DecisionTreeClassifier(),
    "Naive Bayes": lambda: MultinomialNB()
}


# ==========================
# TRAIN MODELS
# ==========================
tf_models = {}
tfidf_models = {}

for name, model_func in models.items():
    tf_models[name] = model_func().fit(X_tf, y)
    tfidf_models[name] = model_func().fit(X_tfidf, y)


# ==========================
# STREAMLIT UI
# ==========================
st.title("Hindi Sentiment Analysis System 🇮🇳")

st.write("TF vs TF-IDF using Multiple Machine Learning Algorithms")

text = st.text_area("Enter Hindi Text")


# ==========================
# PREDICT
# ==========================
if st.button("Predict"):

    tf_input = tf_vectorizer.transform([text])
    tfidf_input = tfidf_vectorizer.transform([text])

    results = []

    for name in models.keys():

        tf_pred = tf_models[name].predict(tf_input)[0]
        tfidf_pred = tfidf_models[name].predict(tfidf_input)[0]

        tf_sent = "Positive 😊" if tf_pred == 1 else "Negative 😡"
        tfidf_sent = "Positive 😊" if tfidf_pred == 1 else "Negative 😡"

        results.append([name, tf_sent, tfidf_sent])


    # ==========================
    # TABLE OUTPUT
    # ==========================
    df = pd.DataFrame(
        results,
        columns=["Algorithm", "TF", "TF-IDF"]
    )

    st.subheader("Prediction Results")
    st.table(df)


    # ==========================
    # GRAPH
    # ==========================
    st.subheader("TF vs TF-IDF Comparison Graph")

    tf_values = [1 if "Positive" in x else 0 for x in df["TF"]]
    tfidf_values = [1 if "Positive" in x else 0 for x in df["TF-IDF"]]

    x = np.arange(len(df["Algorithm"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(x - width/2, tf_values, width, label="TF")
    ax.bar(x + width/2, tfidf_values, width, label="TF-IDF")

    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Sentiment")
    ax.set_title("TF vs TF-IDF Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], rotation=30)

    ax.legend()

    st.pyplot(fig)