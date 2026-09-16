import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, customer_texts, replies, max_features=25000):
        self.customer_texts = customer_texts
        self.replies = replies
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1,2), min_df=2, max_features=max_features
        )
        self.matrix = self.vectorizer.fit_transform(customer_texts)

    def search(self, query, k=3):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [
            {"historical_customer": self.customer_texts[i],
             "historical_reply": self.replies[i],
             "similarity": float(sims[i])}
            for i in idx
        ]

def clean_reply(text):
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()
