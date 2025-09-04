import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import pandas as pd
import numpy as np

class NewsImageTextDataset(Dataset):
    def __init__(self, csv_path, preprocess):
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.data.iloc[idx]['image_path']))
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        return img, text, label

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare dataset and dataloader
dataset = NewsImageTextDataset("D:/Project/Hackathon/dataset/multimodal.csv", preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(5):
    model.train()
    for images, texts, labels in dataloader:
        images = images.to(device)
        text_tokens = clip.tokenize(list(texts)).to(device)
        labels = labels.float().to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        # Compute cosine similarity
        logits = (image_features * text_features).sum(dim=1)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f\"Epoch {epoch+1}, Loss: {loss.item():.4f}\")

# Save the model
torch.save(model.state_dict(), 'D:/Project/Hackathon/models/clip_multimodal.pth')