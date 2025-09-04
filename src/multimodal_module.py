import torch
import clip
from PIL import Image
import numpy as np
import os

class MultimodalModule:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def compute_similarity(self, image_path, text):
        try:
            # Open and preprocess the image.
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error processing image: {e}")
            return 0.5  # Default similarity if image fails to load
        # Tokenize the text.
        text_input = clip.tokenize([text]).to(self.device)
        # Compute image and text features without gradient computation.
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_input)
        # Compute cosine similarity between image and text features.
        similarity = torch.cosine_similarity(image_features, text_features).item()
        return similarity

    def predict(self, image_path, text):
        similarity = self.compute_similarity(image_path, text)
        # Convert similarity to a fake news probability:
        # Lower similarity means less alignment between image and text (more likely fake).
        fake_prob = 1 - similarity
        return np.clip(fake_prob, 0, 1)

    def batch_predict(self, image_paths, texts):
        # For batch evaluation on a dataset
        results = []
        for img, txt in zip(image_paths, texts):
            results.append(self.predict(img, txt))
        return results

if __name__ == "__main__":
    mm = MultimodalModule()
    test_text = "Breaking: NASA confirms Mars colony project will start in 2025."
    # Make sure this image path points to a valid image file on your system.
    image_path = r"D:\Coding\Research_work\data\images\sample.jpg"
    score = mm.predict(image_path, test_text)
    print("Multimodal Module Score:", score)
