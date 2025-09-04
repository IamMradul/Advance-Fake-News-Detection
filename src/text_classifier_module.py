import re
import joblib
import pandas as pd
import numpy as np

class TextClassifierModule:
    def __init__(self, model_path='D:/Project/Hackathon/models/text_clf.joblib', vectorizer_path='D:/Project/Hackathon/models/tfidf_vec.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        cleaned = self.clean_text(text)
        X = self.vectorizer.transform([cleaned])
        prob_fake = self.model.predict_proba(X)[0][0]  # Probability of 'fake' (label 0)
        return prob_fake 