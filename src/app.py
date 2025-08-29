import streamlit as st
from speech_to_text import convert_audio_to_text
from text_preprocessing import preprocess_text
from sentiment_analysis import SentimentAnalyzer
from emotion_analysis import EmotionAnalyzer
from tone_analysis import ToneAnalyzer
from red_green_flag import FlagDetector
import os

# Page configuration
st.set_page_config(
    page_title="Brave Haven",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ BRAVE HAVEN")
st.markdown("**Analyze text and voice input for behavioral red/green flags using advanced AI**")

# Initialize analyzers (with caching to improve performance)
@st.cache_resource
def load_analyzers():
    return {
        'sentiment': SentimentAnalyzer(),
        'emotion': EmotionAnalyzer(),
        'tone': ToneAnalyzer(),
        'flag': FlagDetector()
    }

analyzers = load_analyzers()

# Sidebar for sample texts and information
st.sidebar.header("📝 Sample Texts for Testing")

sample_texts = {
    "Positive Example": "I am so happy and excited about this wonderful opportunity! This is absolutely amazing and I feel confident about the future!",
    "Negative Example": "I hate this stupid thing! It's absolutely terrible and makes me furious! This is completely unacceptable!",
    "Neutral Example": "The weather is nice today. I went to the store and bought some groceries. Everything seems normal.",
    "Mixed Example": "I'm really excited about this project, but I'm also scared that I might fail. It's a wonderful opportunity though!",
    "Aggressive Example": "You are an IDIOT! I can't believe how STUPID this is! I'm going to DESTROY anyone who gets in my way!"
}

selected_sample = st.sidebar.selectbox("Choose a sample text:", ["None"] + list(sample_texts.keys()))

if selected_sample != "None":
    st.sidebar.write(f"**{selected_sample}:**")
    st.sidebar.write(sample_texts[selected_sample])

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About This Project")
st.sidebar.write("""
This AI system analyzes human communication to identify potential behavioral patterns:

- **🎯 Sentiment Analysis**: Positive, negative, or neutral
- **😊 Emotion Detection**: Joy, anger, fear, sadness, etc.
- **🎭 Tone Analysis**: Aggressive, confident, tentative, etc.
- **🚩 Flag Detection**: Red flags (concerning) or green flags (positive)
""")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    if selected_sample != "None":
        text_input = st.text_area("Enter text here:", value=sample_texts[selected_sample], height=150)
    else:
        text_input = st.text_area("Enter text here:", height=150)

    # Audio file upload
    audio_file = st.file_uploader("Or upload an audio file (WAV, MP3, M4A):", type=["wav", "mp3", "m4a"])

with col2:
    st.markdown("### 🎯 Quick Actions")
    analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    # if st.button("🧹 Clear Text", use_container_width=True):
    #     st.experimental_rerun()

# Analysis section
if analyze_button: 
    input_text = ""
    
    # Handle audio file
    if audio_file:
        with st.spinner("🎵 Converting audio to text..."):
            # Save the uploaded audio file temporarily
            audio_path = os.path.join("temp_audio", audio_file.name)
            os.makedirs("temp_audio", exist_ok=True)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            
            input_text = convert_audio_to_text(audio_path)
            st.success(f"**Transcribed Text:** {input_text}")
            os.remove(audio_path) # Clean up temp file
    
    # Handle text input
    elif text_input:
        input_text = text_input

    if input_text:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Create tabs for organized display
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Preprocessing", "😊 Sentiment", "🎭 Emotion", "🎯 Tone", "🚩 Flags", "📋 Summary"])
        
        with tab1:
            st.subheader("Text Preprocessing")
            with st.spinner("Processing text..."):
                processed_data = preprocess_text(input_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.info(processed_data["original_text"])
                
            with col2:
                st.write("**Cleaned Text:**")
                st.info(processed_data["cleaned_text"])
            
            st.write("**Processed Tokens (stopwords removed, lemmatized):**")
            st.write(processed_data["tokens"])

        with tab2:
            st.subheader("Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sentiment_scores = analyzers['sentiment'].analyze_sentiment(processed_data["cleaned_text"])
                sentiment_label = analyzers['sentiment'].get_sentiment_label(sentiment_scores["compound"])
            
            # Display sentiment with color coding
            if sentiment_label == "Positive":
                st.success(f"**Sentiment:** {sentiment_label} (Score: {sentiment_scores['compound']:.2f})")
            elif sentiment_label == "Negative":
                st.error(f"**Sentiment:** {sentiment_label} (Score: {sentiment_scores['compound']:.2f})")
            else:
                st.info(f"**Sentiment:** {sentiment_label} (Score: {sentiment_scores['compound']:.2f})")
            
            # Create a simple bar chart for sentiment scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", f"{sentiment_scores['pos']:.2f}")
            with col2:
                st.metric("Neutral", f"{sentiment_scores['neu']:.2f}")
            with col3:
                st.metric("Negative", f"{sentiment_scores['neg']:.2f}")

        with tab3:
            st.subheader("Emotion Analysis")
            with st.spinner("Detecting emotions..."):
                emotion_result = analyzers['emotion'].analyze_emotion(processed_data["cleaned_text"])
            
            emotion_label = "neutral"  # Default
            if emotion_result:
                emotion_label = emotion_result["label"]
                
                # Display emotion with appropriate emoji
                emotion_emojis = {
                    "joy": "😊", "anger": "😠", "fear": "😨", 
                    "sadness": "😢", "surprise": "😮", "disgust": "🤢", 
                    "neutral": "😐"
                }
                emoji = emotion_emojis.get(emotion_label, "😐")
                
                st.success(f"**Dominant Emotion:** {emoji} {emotion_result['label'].title()} (Confidence: {emotion_result['score']:.2f})")
                
                # Progress bar for confidence
                st.progress(emotion_result['score'])
            else:
                st.warning("Could not detect emotion.")

        with tab4:
            st.subheader("Tone Analysis")
            with st.spinner("Analyzing tone..."):
                tone_result = analyzers['tone'].analyze_tone(processed_data["cleaned_text"])
            
            dominant_tone = tone_result["dominant_tone"]
            
            # Display tone with color coding
            tone_colors = {
                "aggressive": "🔴", "confident": "🟢", "tentative": "🟡",
                "analytical": "🔵", "emotional": "🟠", "neutral": "⚪"
            }
            color = tone_colors.get(dominant_tone, "⚪")
            
            st.info(f"**Dominant Tone:** {color} {dominant_tone.title()}")
            
            # Display tone scores
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tone Scores:**")
                for tone, score in tone_result.items():
                    if tone not in ["dominant_tone", "punctuation_analysis"] and isinstance(score, (int, float)):
                        st.write(f"- {tone.title()}: {score}")
            
            with col2:
                st.write("**Punctuation Analysis:**")
                punct = tone_result["punctuation_analysis"]
                st.write(f"- Exclamations: {punct['exclamations']}")
                st.write(f"- Questions: {punct['questions']}")
                st.write(f"- CAPS words: {punct['caps_words']}")

        with tab5:
            st.subheader("Red/Green Flag Detection")
            with st.spinner("Detecting flags..."):
                flag_result = analyzers['flag'].detect_flags(sentiment_label, emotion_label, dominant_tone)
            
            # Display overall status with prominent styling
            status = flag_result["overall_status"]
            if status == "Red Flag":
                st.error(f"🚩 **{status}** - Potential concerns detected")
            elif status == "Green Flag":
                st.success(f"✅ **{status}** - Positive indicators detected")
            elif status == "Mixed (Both Red and Green Flags)":
                st.warning(f"⚠️ **{status}** - Mixed signals detected")
            else:
                st.info(f"ℹ️ **{status}** - No significant flags detected")
            
            # Display flags in columns
            col1, col2 = st.columns(2)
            
            with col1:
                if flag_result["red_flags"]:
                    st.write("**🚩 Red Flags:**")
                    for flag in flag_result["red_flags"]:
                        st.error(f"• {flag}")
            
            with col2:
                if flag_result["green_flags"]:
                    st.write("**✅ Green Flags:**")
                    for flag in flag_result["green_flags"]:
                        st.success(f"• {flag}")

        with tab6:
            st.subheader("Summary Report")
            
            # Create a summary card
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            """, unsafe_allow_html=True)
            
            st.write("**📝 Input Text:**")
            st.write(f"_{input_text}_")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("😊 Sentiment", sentiment_label)
            with col2:
                st.metric("🎭 Emotion", emotion_label.title())
            with col3:
                st.metric("🎯 Tone", dominant_tone.title())
            with col4:
                st.metric("🚩 Assessment", status.split()[0])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if status == "Red Flag":
                st.warning("""
                **Recommendations for Red Flag Detection:**
                - Consider professional counseling or support
                - Practice stress management techniques
                - Engage in positive communication exercises
                - Seek feedback from trusted friends or colleagues
                """)
            elif status == "Green Flag":
                st.success("""
                **Great Communication Detected:**
                - Continue with this positive communication style
                - You're expressing yourself clearly and positively
                - This type of communication builds trust and rapport
                """)
            else:
                st.info("""
                **Neutral Communication:**
                - Communication appears balanced
                - Consider adding more positive elements if appropriate
                - Monitor for any changes in communication patterns
                """)

    else: 
        st.warning("⚠️ Please enter some text or upload an audio file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p>🛡️ Brave Haven | DEVELOP BY OSAMA AHMED | Built with Streamlit, NLTK, and Hugging Face Transformers</p>
<p><small>This tool is for educational and demonstration purposes. For serious mental health concerns, please consult a professional.</small></p>
</div>
""", unsafe_allow_html=True)

