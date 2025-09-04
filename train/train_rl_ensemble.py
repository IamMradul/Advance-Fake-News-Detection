# save as train/train_rl_ensemble.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
import pandas as pd

# 1. Load validation data
true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use a validation split (e.g., 20%)
from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Load all modules (adjust imports as needed)
from src.stylometry_module import StylometryModule
from src.metadata_module import MetadataAnalysisModule
from src.gnn_module import GNNModule
from src.temporal_module import TemporalConsistencyModule
from src.text_classifier_module import TextClassifierModule
from src.transformer_module import TransformerModule

modules = [
    TextClassifierModule(),
    StylometryModule(),
    MetadataAnalysisModule(),
    GNNModule(),
    TemporalConsistencyModule(),
    TransformerModule(),
    # Add more modules if you have them
]

# 3. Get predictions from each module
X_val = val_df  # Should be the full validation set
print(f"Number of validation samples: {len(X_val)}")
y_val = val_df['label'].values

probas = []
for i, module in enumerate(modules):
    if hasattr(module, "predict_proba"):
        probas.append(module.predict_proba(X_val))
    else:
        if isinstance(X_val, pd.DataFrame) and 'content' in X_val.columns:
            texts = X_val['content']
        else:
            texts = X_val
        print(f"Number of texts for module {module.__class__.__name__}: {len(texts)}")
        module_probs = []
        for idx, text in enumerate(texts):
            # Special handling for MetadataAnalysisModule
            if module.__class__.__name__ == 'MetadataAnalysisModule':
                row = X_val.iloc[idx]
                metadata_input = {
                    'subject': row['subject'] if 'subject' in row else 'unknown',
                    'domain': row['domain'] if 'domain' in row else 'unknown'
                }
                prob_fake = module.predict(metadata_input)
            # Special handling for GNNModule
            elif module.__class__.__name__ == 'GNNModule':
                row = X_val.iloc[idx]
                subject = row['subject'] if 'subject' in row else 'unknown'
                prob_fake = module.predict(text, subject)
            else:
                prob_fake = module.predict(text)
            module_probs.append([prob_fake, 1 - prob_fake])
        probas.append(np.array(module_probs))

for i, p in enumerate(probas):
    print(f"Module {i} output shape: {p.shape}")
print(f"y_val shape: {y_val.shape}")

probas = np.stack(probas, axis=0)  # shape: (num_modules, num_samples, 2)

# 4. Optimize ensemble weights (random search)
best_acc = 0
best_weights = None

for _ in range(1000):
    weights = np.random.dirichlet(np.ones(len(modules)), size=1)[0]
    # Weighted sum of probabilities
    ensemble_proba = np.tensordot(weights, probas, axes=([0], [0]))
    preds = np.argmax(ensemble_proba, axis=1)
    acc = (preds == y_val).mean()
    if acc > best_acc:
        best_acc = acc
        best_weights = weights

print("Best validation accuracy:", best_acc)
print("Best weights:", best_weights)

# 5. Save the best weights
np.save('D:/Project/Hackathon/models/rl_ensemble_weights.npy', best_weights)