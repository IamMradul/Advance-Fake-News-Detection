# train_metadata.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load ISOT dataset
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use 'subject' as a metadata feature (proxy for domain/source)
X_raw = df[['subject']].fillna('unknown')
y = df['label'].values

# One-hot encode the subject
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = encoder.fit_transform(X_raw)

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

# Save model and encoder
joblib.dump(clf, 'D:/Project/Hackathon/models/metadata_clf.joblib')
joblib.dump(encoder, 'D:/Project/Hackathon/models/metadata_encoder.joblib')