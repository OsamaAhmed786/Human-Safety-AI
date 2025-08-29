from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)

    def analyze_emotion(self, text):
        """
        Analyzes the emotion of the given text using a pre-trained Hugging Face model.
        Returns a list of dictionaries, each containing 'label' and 'score'.
        """
        if not text or not isinstance(text, str):
            return []
        
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'joy', 'score': 0.99}]
        # We are interested in the top emotion.
        results = self.classifier(text)
        return results[0] if results else None

if __name__ == '__main__':
    analyzer = EmotionAnalyzer()

    texts = [
        "I am so happy today!",
        "This is incredibly frustrating and I feel angry.",
        "I am feeling very sad about the news.",
        "What a surprise, I did not expect that!",
        "I am so scared right now.",
        "This is a neutral statement."
    ]

    for text in texts:
        emotion = analyzer.analyze_emotion(text)
        print(f"Text: \"{text}\"\nEmotion: {emotion}\n")


