import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 1. Load ISOT dataset
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)

# 2. Prepare sequences
N = 3  # sequence length
sequences = []
labels = []
for subject, group in df.groupby('subject'):
    group = group.sort_values('date') if 'date' in group.columns else group
    texts = group['text'].tolist()
    group_labels = group['label'].tolist()
    for i in range(len(texts) - N + 1):
        seq = texts[i:i+N]
        label = group_labels[i+N-1]
        sequences.append(seq)
        labels.append(label)

# 3. Vectorize text (TF-IDF, then average for each sequence position)
vectorizer = TfidfVectorizer(max_features=100)
all_texts = [t for seq in sequences for t in seq]
vectorizer.fit(all_texts)

def seq_to_vec(seq):
    return np.stack([vectorizer.transform([t]).toarray()[0] for t in seq])

X = np.stack([seq_to_vec(seq) for seq in sequences])
y = np.array(labels)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SequenceDataset(X_train, y_train)
test_ds = SequenceDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

# 6. Bi-LSTM Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMClassifier(input_dim=100, hidden_dim=64, num_layers=1, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 7. Training loop
for epoch in range(1, 11):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct / total:.4f}")

# 8. Save the model and vectorizer
torch.save(model.state_dict(), 'D:/Project/Hackathon/models/temporal_bilstm.pth')
import joblib
joblib.dump(vectorizer, 'D:/Project/Hackathon/models/temporal_vectorizer.joblib')

print(Counter(labels))

print(sequences[:3])
print(labels[:3])

print(len(set(tuple(seq) for seq in sequences)), len(sequences))