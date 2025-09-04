# train_gnn.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
import joblib

# 1. Load and label the ISOT dataset
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Use both title and text if available, else just text
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text'].fillna('')
elif 'content' in df.columns:
    df['content'] = df['content'].fillna('')
else:
    raise ValueError("No suitable text column found in the dataset.")

# 3. Build node features (TF-IDF vectors)
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['label'].values

joblib.dump(vectorizer, 'D:/Project/Hackathon/models/gnn_vectorizer.joblib')

# 4. Build edges: connect articles with the same subject
subject_to_indices = {}
for idx, subject in enumerate(df['subject']):
    subject_to_indices.setdefault(subject, []).append(idx)

edge_index = []
for indices in subject_to_indices.values():
    if len(indices) > 1:
        for i in indices:
            # Connect to up to 3 other random articles with the same subject
            others = [j for j in indices if j != i]
            random.shuffle(others)
            for j in others[:3]:
                edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

# 5. Create PyTorch Geometric Data object
data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long)
)

# 6. Train/test split (use masks)
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_indices, test_indices = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)
train_mask[train_indices] = True
test_mask[test_indices] = True
data.train_mask = train_mask
data.test_mask = test_mask

# 7. Define a simple GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=X.shape[1], hidden_dim=32, num_classes=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 8. Training loop
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        _, pred = out[data.test_mask].max(dim=1)
        correct = pred.eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')

# 9. Save the model
torch.save(model.state_dict(), 'D:/Project/Hackathon/models/gnn_model.pth')