# src/gnn_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import joblib
import numpy as np
# from torch_geometric.nn import GCNConv  # Uncomment if you have torch-geometric installed

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GNNModule:
    def __init__(self, model_path='D:/Project/Hackathon/models/gnn_model.pth', vectorizer_path='D:/Project/Hackathon/models/gnn_vectorizer.joblib'):
        # Load the trained GNN model and TF-IDF vectorizer
        self.vectorizer = joblib.load(vectorizer_path)
        self.num_features = self.vectorizer.transform(["test"]).shape[1]
        self.model = GCN(num_features=self.num_features, hidden_dim=32, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def build_graph_for_article(self, text, subject, subject_to_indices=None, idx=0):
        # Build a minimal graph for a single article (node 0)
        x = torch.tensor(self.vectorizer.transform([text]).toarray(), dtype=torch.float)
        # For a single node, edge_index is empty
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        return data

    def predict(self, text, subject):
        data = self.build_graph_for_article(text, subject)
        with torch.no_grad():
            out = self.model(data)
            prob_fake = F.softmax(out, dim=1)[0][0].item()  # Probability for class 0 (fake)
        return np.clip(prob_fake, 0, 1)

# Test:
if __name__ == "__main__":
    # Example usage
    gnn = GNNModule(
        model_path='D:/Project/Hackathon/models/gnn_model.pth',
        vectorizer_path='D:/Project/Hackathon/models/gnn_vectorizer.joblib'
    )
    test_text = "Breaking: NASA confirms Mars colony project will start in 2025."
    test_subject = "space"
    print("GNN Module Score:", gnn.predict(test_text, test_subject))
