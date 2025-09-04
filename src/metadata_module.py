import numpy as np
import joblib


class MetadataAnalysisModule:
    def __init__(self, model_path='D:/Project/Hackathon/models/metadata_clf.joblib', encoder_path='D:/Project/Hackathon/models/metadata_encoder.joblib'):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

    def predict(self, metadata):
        """
        metadata: dict with keys like 'subject', etc.
        Returns a fake probability (float between 0 and 1).
        """
        # Prepare input for encoder
        X_raw = [[metadata.get('subject', 'unknown')]]
        X_enc = self.encoder.transform(X_raw)
        prob_fake = self.model.predict_proba(X_enc)[0][0]
        return np.clip(prob_fake, 0, 1)

if __name__ == "__main__":
    mam = MetadataAnalysisModule()
    print("Politics subject:", mam.predict({'subject': 'politics'}))
    print("Unknown subject:", mam.predict({'subject': 'unknown'})) 