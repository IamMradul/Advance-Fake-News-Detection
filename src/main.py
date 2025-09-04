# src/main.py

import requests
import os
from dotenv import load_dotenv
from transformer_module import TransformerModule
from questioning_module import QuestioningModule
from stylometry_module import StylometryModule
from gnn_module import GNNModule
from multimodal_module import MultimodalModule
from rl_module import RLModule
from ensemble import EnsembleModule
from text_classifier_module import TextClassifierModule
from temporal_module import TemporalConsistencyModule
from metadata_module import MetadataAnalysisModule
from nli_module import NLIModule
from feedback_module import FeedbackModule

# Load environment variables from .env in src directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

def check_fact_api(article_text):
    """
    Dummy function to simulate checking a fact-checking API.
    Replace the URL and payload with those required by a real API.
    """
    # Set your .env file with: FACT_API_KEY=your_actual_key
    API_KEY = os.getenv('FACT_API_KEY', 'YOUR_FACT_API_KEY')
    
    news_text = article_text  # or preprocess as needed
    params = {"query": news_text, "key": API_KEY}
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            # You can process 'result' to extract relevant information
            # For example, check if any claims are returned
            claims = result.get("claims", [])
            # Return a fake probability based on whether claims were found
            return 0.0 if claims else 0.5  # 0.0 = likely real, 0.5 = unknown
        else:
            print("API response code:", response.status_code)
            return 0.5
    except Exception as e:
        print("API check error:", e)
        return 0.5

def main():
    print("Welcome to the Advanced Fake News Detection System!")
    
    # Request user input
    article_text = input("Enter the news article text:\n")
    trusted_context = input("Enter a trusted context for fact-checking (or press Enter to use the article text):\n")
    if not trusted_context.strip():
        trusted_context = article_text
    image_path = input("Enter the full path to the image associated with the news (or press Enter if not available):\n")
    if not image_path.strip():
        image_path = None  # We'll use a default value later if needed

    # Initialize modules
    text_clf = TextClassifierModule(
        model_path='D:/Project/Hackathon/models/text_clf.joblib',
        vectorizer_path='D:/Project/Hackathon/models/tfidf_vec.joblib'
    )
    #transformer = TransformerModule()
    questioning = QuestioningModule()
    stylometry = StylometryModule()
    gnn = GNNModule()
    multimodal = MultimodalModule() if image_path else None
    #rl = RLModule()
    temporal = TemporalConsistencyModule()
    metadata = MetadataAnalysisModule()
    nli = NLIModule()
    feedback = FeedbackModule()
    
    # Get predictions from each module
    text_clf_prob = text_clf.predict(article_text)
    #transformer_prob = transformer.predict(article_text)
    questioning_prob = questioning.predict(article_text)
    stylometry_prob = stylometry.predict(article_text)
    # If you have a subject variable, use it; otherwise, use 'unknown'
    subject = "unknown"  # Or extract from metadata if available
    gnn_prob = gnn.predict(article_text, subject)
    if multimodal:
        multimodal_prob = multimodal.predict(image_path, article_text)
    else:
        multimodal_prob = 0.5
    #rl_prob = rl.predict()
    temporal_prob = temporal.predict([article_text])
    # Simulate metadata input (in real use, parse from article)
    metadata_input = {'domain': 'bbc.com'}
    metadata_prob = metadata.predict(metadata_input)
    nli_prob = nli.predict(article_text, trusted_context)
    api_prob = check_fact_api(article_text)
    
    # Print individual module outputs
    print("\nModule Predictions:")
    print("Text Classifier Score:", text_clf_prob)
    #print("Transformer Score:", transformer_prob)
    print("Questioning Score:", questioning_prob)
    print("Stylometry Score:", stylometry_prob)
    print("GNN Score:", gnn_prob)
    print("Multimodal Score:", multimodal_prob)
    # print("RL Score:", rl_prob)
    print("Temporal Consistency Score:", temporal_prob)
    print("Metadata Analysis Score:", metadata_prob)
    print("NLI Score:", nli_prob)
    print("API Score:", api_prob)
    
    # Update ensemble weights to accommodate all 12 modules.
    ensemble = EnsembleModule(weights=[0.13, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07])
    module_outputs = [text_clf_prob, questioning_prob, stylometry_prob, gnn_prob, multimodal_prob, temporal_prob, metadata_prob, nli_prob, api_prob]
    ensemble_prob, final_label = ensemble.predict(module_outputs)
    
    print("\nFinal Fake News Probability: {:.2f}".format(ensemble_prob))
    print("Final Classification:", final_label)
    
    # Collect user feedback
    user_label = input("\nDo you think this article is Real or Fake? (Enter 'Real' or 'Fake'): ")
    feedback.collect_feedback(article_text, final_label, user_label)

if __name__ == "__main__":
    main()
