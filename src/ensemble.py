# src/ensemble.py
class EnsembleModule:
    def __init__(self, weights=None):
        # Define default weights for each module's output.
        # Adjust these weights based on the performance and reliability of each module.
        if weights is None:
            self.weights = [0.3, 0.15, 0.15, 0.15, 0.15, 0.1]  # Example for transformer, questioning, stylometry, GNN, multimodal, RL
        else:
            self.weights = weights

    def predict(self, outputs):
        ensemble_prob = sum(w * p for w, p in zip(self.weights, outputs))
        final_label = "Fake" if ensemble_prob > 0.5 else "Real"
        return ensemble_prob, final_label

if __name__ == "__main__":
    ensemble = EnsembleModule()
    sample_outputs = [0.95, 1.0, 0.95, 0.90, 0.85, 0.93]  # simulated scores from each module
    prob, label = ensemble.predict(sample_outputs)
    print("Ensemble Score:", prob, "Final Label:", label)

