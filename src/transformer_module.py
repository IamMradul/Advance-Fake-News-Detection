from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

class TransformerModule:
    def __init__(self, model_path='D:/Project/Hackathon/bert_fakenews'):
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # Assuming label 0 is 'fake', return its probability
        return float(np.clip(probs[0], 0, 1))

if __name__ == "__main__":
    tm = TransformerModule(model_path='D:/Project/Hackathon/bert_fakenews')
    test_text = "Breaking: NASA confirms Mars colony project will start in 2025."
    print("Transformer Module Score:", tm.predict(test_text))
