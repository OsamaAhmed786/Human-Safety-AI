from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of the given text using VADER.
        Returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
        """
        if not text or not isinstance(text, str):
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        
        vs = self.analyzer.polarity_scores(text)
        return vs

    def get_sentiment_label(self, compound_score, threshold=0.05):
        """
        Returns a sentiment label (positive, negative, neutral) based on the compound score.
        """
        if compound_score >= threshold:
            return "Positive"
        elif compound_score <= -threshold:
            return "Negative"
        else:
            return "Neutral"

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()

    texts = [
        "This is a wonderful movie!",
        "I hate this product, it's terrible.",
        "The weather is okay today.",
        "I am so happy and excited!",
        "This is absolutely disgusting and unacceptable."
    ]

    for text in texts:
        scores = analyzer.analyze_sentiment(text)
        label = analyzer.get_sentiment_label(scores["compound"])
        print(f"Text: \"{text}\"\nScores: {scores}\nLabel: {label}\n")


