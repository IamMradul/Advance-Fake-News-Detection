import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Load the ISOT Fake News Dataset
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')

# 2. Add labels: 1 for Real, 0 for Fake
true_df['label'] = 1
fake_df['label'] = 0

# 3. Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Preprocess the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Use both title and text if available, else just text
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text'].fillna('')
elif 'content' in df.columns:
    df['content'] = df['content'].fillna('')
else:
    raise ValueError("No suitable text column found in the dataset.")

df['content'] = df['content'].apply(clean_text)

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], test_size=0.2, random_state=42
)

# 6. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 8. Predict and evaluate
y_pred = clf.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the trained model and vectorizer for integration
joblib.dump(clf, 'D:/Project/Hackathon/models/text_clf.joblib')
joblib.dump(vectorizer, 'D:/Project/Hackathon/models/tfidf_vec.joblib') 