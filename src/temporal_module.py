import torch
import torch.nn as nn
import numpy as np
import joblib

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h)

class TemporalConsistencyModule:
    def __init__(self, model_path='D:/Project/Hackathon/models/temporal_bilstm.pth', vectorizer_path='D:/Project/Hackathon/models/temporal_vectorizer.joblib', seq_len=3):
        self.vectorizer = joblib.load(vectorizer_path)
        self.seq_len = seq_len
        self.input_dim = self.vectorizer.transform(["test"]).shape[1]
        self.model = BiLSTMClassifier(input_dim=self.input_dim, hidden_dim=64, num_layers=1, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def preprocess_sequence(self, text_sequence):
        # Pad or truncate sequence to seq_len
        seq = text_sequence[-self.seq_len:]
        if len(seq) < self.seq_len:
            seq = [""] * (self.seq_len - len(seq)) + seq
        X = np.stack([self.vectorizer.transform([t]).toarray()[0] for t in seq])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len, input_dim)

    def predict(self, text_sequence):
        x = self.preprocess_sequence(text_sequence)
        with torch.no_grad():
            out = self.model(x)
            prob_fake = torch.softmax(out, dim=1)[0][0].item()  # Probability for class 0 (fake)
        return np.clip(prob_fake, 0, 1)

if __name__ == "__main__":
    tcm = TemporalConsistencyModule(
        model_path='D:/Project/Hackathon/models/temporal_bilstm.pth',
        vectorizer_path='D:/Project/Hackathon/models/temporal_vectorizer.joblib',
        seq_len=3
    )
    sequence = [
        "Article 1 text here.",
        "Article 2 text here.",
        "Article 3 text here."
    ]
    print("Temporal Consistency Module Score:", tcm.predict(sequence)) 