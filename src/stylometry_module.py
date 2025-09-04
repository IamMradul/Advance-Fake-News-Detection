import nltk
import textstat
import numpy as np
import joblib

# Ensure the necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # Add this line

import random

class StylometryModule:
    def __init__(self, model_path='D:/Project/Hackathon/models/stylometry_clf.joblib'):
        self.model = joblib.load(model_path)

    def extract_features(self, text):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            avg_sentence_length = 0
            total_tokens = 0
            noun_count = 0
            verb_count = 0
        else:
            total_tokens = 0
            noun_count = 0
            verb_count = 0
            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                total_tokens += len(tokens)
                pos_tags = nltk.pos_tag(tokens)
                for _, tag in pos_tags:
                    if tag.startswith("NN"):
                        noun_count += 1
                    elif tag.startswith("VB"):
                        verb_count += 1
            avg_sentence_length = total_tokens / len(sentences)
        noun_ratio = noun_count / total_tokens if total_tokens > 0 else 0
        verb_ratio = verb_count / total_tokens if total_tokens > 0 else 0
        return [
            textstat.flesch_reading_ease(text),
            textstat.smog_index(text),
            avg_sentence_length,
            noun_ratio,
            verb_ratio
        ]

    def predict(self, text):
        features = np.array(self.extract_features(text)).reshape(1, -1)
        # Predict probability for class 0 (fake)
        prob_fake = self.model.predict_proba(features)[0][0]
        return np.clip(prob_fake, 0, 1)

if __name__ == "__main__":
    sm = StylometryModule()
    test_text = "Breaking: NASA confirms Mars colony project will start in 2025."
    print("Stylometry Module Score:", sm.predict(test_text))
