# src/rl_module.py
import random
import numpy as np

class RLModule:
    def __init__(self, weights_path='D:/Project/Hackathon/models/rl_ensemble_weights.npy'):
        self.weights = np.load(weights_path)

    def predict(self, module_outputs):
        # module_outputs: list or np.array of module probabilities for one article
        score = np.dot(self.weights, module_outputs)
        return float(np.clip(score, 0, 1))

# Test:
if __name__ == "__main__":
    rl = RLModule()
    sample_outputs = [0.8, 0.7, 0.6, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4]  # Example module outputs
    print("RL Module Ensemble Score:", rl.predict(sample_outputs))

# --- Training script for RL ensemble weights (random search) ---
# Save this as train_rl_ensemble.py and run it after saving module outputs and labels
'''
import numpy as np
from sklearn.metrics import accuracy_score

# Load your validation set module outputs and true labels
# Each row in module_outputs is a list of module probabilities for one article
module_outputs = np.load('D:/Project/Hackathon/models/ensemble_val_outputs.npy')  # shape: (num_samples, num_modules)
labels = np.load('D:/Project/Hackathon/models/ensemble_val_labels.npy')  # shape: (num_samples,)

def evaluate(weights):
    scores = np.dot(module_outputs, weights)
    preds = (scores > 0.5).astype(int)
    return accuracy_score(labels, preds)

best_acc = 0
best_weights = None
for _ in range(10000):
    w = np.random.dirichlet(np.ones(module_outputs.shape[1]))
    acc = evaluate(w)
    if acc > best_acc:
        best_acc = acc
        best_weights = w
        print(f"New best acc: {best_acc:.4f}, weights: {best_weights}")

# Save the best weights
np.save('D:/Project/Hackathon/models/rl_ensemble_weights.npy', best_weights)
'''
