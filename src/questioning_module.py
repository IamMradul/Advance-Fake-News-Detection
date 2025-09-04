# src/questioning_module.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env in src directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class QuestioningModule:
    def __init__(self, api_key=None):
        # Set your .env file with: NEWS_API_KEY=your_actual_key
        self.api_key = api_key or os.getenv('NEWS_API_KEY', 'YOUR_NEWS_API_KEY')
        self.api_url = 'https://newsapi.org/v2/everything'

    def generate_questions(self, headline):
        # Generate 4W questions from the headline
        questions = [
            f"Who is involved in: {headline}?",
            f"What happened in: {headline}?",
            f"When did it happen in: {headline}?",
            f"Where did it happen in: {headline}?"
        ]
        return questions

    def ask_news_api(self, question):
        # Query the NewsAPI for the question (simulate for now)
        # In a real implementation, use the API to search for relevant articles
        params = {
            'q': question,
            'apiKey': self.api_key,
            'pageSize': 1
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # If at least one article is found, consider it a relevant answer
                return len(data.get('articles', [])) > 0
            else:
                return False
        except Exception as e:
            print(f"API error for question '{question}':", e)
            return False

    def predict(self, headline):
        questions = self.generate_questions(headline)
        relevant_count = 0
        for q in questions:
            if self.ask_news_api(q):
                relevant_count += 1
        # Return the ratio of questions with relevant answers (higher = more likely real)
        return relevant_count / len(questions) if questions else 0.0

if __name__ == "__main__":
    # Make sure you have a .env file in src/ with NEWS_API_KEY=your_actual_key
    qm = QuestioningModule()
    headline = "NASA confirms Mars colony project will start in 2025."
    score = qm.predict(headline)
    print("Questioning Module Relevance Score:", score)
