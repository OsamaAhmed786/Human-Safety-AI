import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean the input text by removing special characters, extra spaces, etc.
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize the text into words using simple split.
        """
        return text.split()
    
    def remove_punctuation(self, tokens):
        """
        Remove punctuation from tokens.
        """
        return [token for token in tokens if token not in string.punctuation]
    
    def remove_stopwords(self, tokens):
        """
        Remove stop words from tokens.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens to their root form.
        """
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    def preprocess(self, text, remove_stopwords=True, lemmatize=True):
        """
        Complete preprocessing pipeline.
        Returns both the cleaned text and processed tokens.
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove punctuation
        tokens = self.remove_punctuation(tokens)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Join tokens back to create processed text
        processed_text = ' '.join(tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'processed_text': processed_text
        }

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Convenience function for quick text preprocessing.
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text, remove_stopwords, lemmatize)

if __name__ == '__main__':
    # Example usage
    sample_text = "Hello! This is a SAMPLE text with some URLs like https://example.com and emails like test@example.com. It has punctuation, stopwords, and needs preprocessing!!!"
    
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess(sample_text)
    
    print("Original Text:", result['original_text'])
    print("Cleaned Text:", result['cleaned_text'])
    print("Tokens:", result['tokens'])
    print("Processed Text:", result['processed_text'])

