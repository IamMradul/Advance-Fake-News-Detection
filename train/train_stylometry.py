# train_stylometry.py

import pandas as pd
import numpy as np
import nltk
import textstat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_features(text):
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

# Load ISOT dataset
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use both title and text if available, else just text
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text'].fillna('')
elif 'content' in df.columns:
    df['content'] = df['content'].fillna('')
else:
    raise ValueError("No suitable text column found in the dataset.")

# Extract features
feature_list = []
for text in df['content']:
    feature_list.append(extract_features(text))
X = np.array(feature_list)
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Save model
joblib.dump(clf, 'D:/Project/Hackathon/models/stylometry_clf.joblib')