class FeedbackModule:
    def __init__(self):
        self.feedback_log = []

    def collect_feedback(self, article_text, prediction, user_label):
        feedback = {
            'article_text': article_text,
            'prediction': prediction,
            'user_label': user_label
        }
        self.feedback_log.append(feedback)
        print("Feedback collected:", feedback)
        # In a real system, use this feedback to retrain models or adjust weights

    def get_feedback_log(self):
        return self.feedback_log

if __name__ == "__main__":
    fm = FeedbackModule()
    fm.collect_feedback("NASA launches Mars mission.", "Fake", "Real")
    print(fm.get_feedback_log()) 