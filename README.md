# Human Safety AI 

## Overview

The Human Safety AI Project is a comprehensive artificial intelligence system designed to analyze human communication patterns and identify potential behavioral red flags or green flags through advanced natural language processing techniques. This project combines multiple AI technologies including sentiment analysis, emotion detection, tone analysis, and speech-to-text conversion to provide a holistic assessment of human communication safety.

## Features

### Core Capabilities

- **Text Analysis**: Comprehensive analysis of written text input
- **Speech-to-Text Conversion**: Convert audio files to text for analysis
- **Sentiment Analysis**: Determine positive, negative, or neutral sentiment using VADER
- **Emotion Detection**: Identify specific emotions using pre-trained Hugging Face models
- **Tone Analysis**: Analyze communication tone (aggressive, confident, tentative, etc.)
- **Red/Green Flag Detection**: Rule-based system to identify concerning or positive patterns

### User Interface

- **Interactive Web Application**: Built with Streamlit for easy use
- **Sample Text Testing**: Pre-loaded examples for quick testing
- **Tabbed Analysis Results**: Organized display of all analysis components
- **Real-time Processing**: Immediate analysis results with visual feedback

## Technology Stack

- **Python 3.11**: Core programming language
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing toolkit
- **Hugging Face Transformers**: Pre-trained machine learning models
- **VADER Sentiment**: Valence Aware Dictionary and sEntiment Reasoner
- **SpeechRecognition**: Audio-to-text conversion
- **PyTorch**: Deep learning framework

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Internet connection for downloading models

### Setup Instructions

1. Clone or download the project files
    ```
    Git clone https://github.com/OsamaAhmed786/Human-Safety-AI.git

2. Navigate to the project directory:
   ```
   cd Human-Safety-AI
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run src/app.py
   ```

5. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

## Usage Guide

### Text Analysis

1. **Enter Text**: Type or paste text into the main text area
2. **Select Sample**: Choose from pre-loaded examples in the sidebar
3. **Click Analyze**: Press the "🔍 Analyze Text" button to start processing
4. **Review Results**: Navigate through the tabbed results to see detailed analysis

### Audio Analysis

1. **Upload Audio**: Click "Browse files" and select a WAV, MP3, or M4A file
2. **Automatic Conversion**: The system will convert speech to text
3. **Analysis**: The converted text will be automatically analyzed
4. **Results**: View the same comprehensive analysis as text input

### Understanding Results

#### Preprocessing Tab
- Shows original and cleaned text
- Displays processed tokens after stopword removal and lemmatization

#### Sentiment Tab
- VADER sentiment scores (positive, negative, neutral, compound)
- Overall sentiment classification with confidence scores

#### Emotion Tab
- Dominant emotion detected using Hugging Face models
- Confidence score for the detected emotion
- Supports: joy, anger, fear, sadness, surprise, disgust, neutral

#### Tone Tab
- Analysis of communication tone patterns
- Scores for: aggressive, confident, tentative, analytical, emotional
- Punctuation analysis (exclamations, questions, caps)

#### Flags Tab
- Red flags: Concerning behavioral patterns
- Green flags: Positive communication indicators
- Overall assessment: Red Flag, Green Flag, Mixed, or Neutral

#### Summary Tab
- Consolidated view of all analysis results
- Recommendations based on detected patterns
- Key metrics and assessment overview

## Project Structure

```
Human-Safety-AI/
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── speech_to_text.py         # Audio processing module
│   ├── text_preprocessing.py     # Text cleaning and tokenization
│   ├── sentiment_analysis.py     # VADER sentiment analysis
│   ├── emotion_analysis.py       # Hugging Face emotion detection
│   ├── tone_analysis.py          # Heuristic tone analysis
│   └── red_green_flag.py         # Flag detection logic
├── temp_audio/                   # Temporary audio file storage
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Technical Details

### Sentiment Analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis, which is specifically designed for social media text and handles:
- Punctuation emphasis
- Capitalization
- Emoticons and slang
- Negation handling

### Emotion Detection
Employs the `j-hartmann/emotion-english-distilroberta-base` model from Hugging Face, which:
- Predicts Ekman's 6 basic emotions plus neutral
- Trained on diverse datasets for robust performance
- Provides confidence scores for predictions

### Tone Analysis
Implements a heuristic-based approach that analyzes:
- Word choice patterns
- Punctuation usage
- Capitalization patterns
- Linguistic markers for different tones

### Flag Detection
Rule-based system that combines:
- Sentiment analysis results
- Emotion detection outcomes
- Tone analysis findings
- Combinatorial pattern recognition

## Educational Value

This project serves as an excellent learning resource for:

### Natural Language Processing Concepts
- Text preprocessing and tokenization
- Sentiment analysis techniques
- Emotion recognition in text
- Feature extraction from linguistic patterns

### Machine Learning Applications
- Pre-trained model usage
- Model fine-tuning concepts
- Ensemble method principles
- Real-world AI application development

### Software Engineering Practices
- Modular code organization
- User interface design
- Error handling and validation
- Documentation and testing

## Limitations and Considerations

### Technical Limitations
- Heuristic tone analysis may not capture subtle nuances
- Limited to English language processing
- Audio quality affects speech-to-text accuracy
- Rule-based flag detection may produce false positives/negatives

### Ethical Considerations
- This tool is for educational and demonstration purposes only
- Should not replace professional mental health assessment
- Privacy concerns with sensitive text analysis
- Potential bias in training data and algorithms

### Performance Considerations
- First-time model loading may take several minutes
- Large audio files may require significant processing time
- Internet connection required for initial model downloads

## Future Enhancements

### Technical Improvements
- Multi-language support
- Advanced neural tone analysis
- Improved audio processing capabilities
- Real-time streaming analysis

### Feature Additions
- Historical analysis tracking
- Customizable flag detection rules
- Integration with external APIs
- Batch processing capabilities

### User Experience
- Mobile-responsive design
- Advanced visualization options
- Export functionality for results
- User preference settings

## Contributing

This project is designed for educational purposes. Potential areas for contribution include:

- Improving accuracy of tone analysis algorithms
- Adding support for additional languages
- Enhancing the user interface design
- Expanding the flag detection rule system
- Adding comprehensive unit tests

## License and Disclaimer

This project is provided for educational and demonstration purposes only. It should not be used as a substitute for professional mental health assessment or intervention. The developers are not responsible for any decisions made based on the analysis results.

For serious mental health concerns, please consult qualified professionals.

## Support and Resources

### Learning Resources
- NLTK Documentation: https://www.nltk.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Streamlit Documentation: https://docs.streamlit.io/
- VADER Sentiment Analysis: https://github.com/cjhutto/vaderSentiment

### Technical Support
- Check the project documentation for common issues
- Review the code comments for implementation details
- Consult the requirements.txt for dependency versions
- Test with the provided sample texts for validation

This comprehensive Human Safety AI project demonstrates the practical application of multiple AI technologies in analyzing human communication patterns, providing valuable insights into sentiment, emotion, and behavioral indicators while serving as an excellent educational resource for understanding modern natural language processing techniques.

📜 License
MIT License © 2025

👨‍💻 Author
Developed by Osama Ahmed 🚀

