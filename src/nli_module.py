from transformers import pipeline
import numpy as np

class NLIModule:
    def __init__(self, model_name='roberta-large-mnli'):
        self.nli = pipeline('text-classification', model=model_name)

    def predict(self, premise, hypothesis):
        result = self.nli(f'{premise} </s> {hypothesis}')[0]
        # 'ENTAILMENT' means supports, 'CONTRADICTION' means likely fake
        if result['label'] == 'CONTRADICTION':
            prob_fake = result['score']
        else:
            prob_fake = 1 - result['score']
        return np.clip(prob_fake, 0, 1)

if __name__ == "__main__":
    nli = NLIModule()
    premise = "NASA announced a Mars mission."
    hypothesis = "NASA denies any Mars mission plans."
    print("NLI Module Score:", nli.predict(premise, hypothesis)) 