class FlagDetector:
    def __init__(self):
        pass

    def detect_flags(self, sentiment_label, emotion_label, dominant_tone):
        """
        Detects red or green flags based on sentiment, emotion, and tone analysis.
        This is a rule-based system for demonstration.
        """
        red_flags = []
        green_flags = []
        
        # Rule 1: Sentiment-based flags
        if sentiment_label == "Negative":
            red_flags.append("Overall negative sentiment detected.")
        elif sentiment_label == "Positive":
            green_flags.append("Overall positive sentiment detected.")

        # Rule 2: Emotion-based flags
        if emotion_label in ["anger", "fear", "disgust", "sadness"]:
            red_flags.append(f"Strong emotion of '{emotion_label}' detected.")
        elif emotion_label == "joy":
            green_flags.append("Emotion of 'joy' detected.")

        # Rule 3: Tone-based flags
        if dominant_tone == "aggressive":
            red_flags.append("Aggressive tone detected.")
        elif dominant_tone == "confident":
            green_flags.append("Confident tone detected.")
        elif dominant_tone == "tentative":
            red_flags.append("Tentative tone detected, indicating uncertainty or hesitation.")
        elif dominant_tone == "emotional":
            red_flags.append("Highly emotional tone detected.")

        # Rule 4: Combinatorial flags (examples)
        if "Overall negative sentiment detected." in red_flags and \
           "Strong emotion of 'anger' detected." in red_flags and \
           "Aggressive tone detected." in red_flags:
            red_flags.append("Critical combination: Negative sentiment, anger, and aggressive tone.")
            
        if "Overall positive sentiment detected." in green_flags and \
           "Emotion of 'joy' detected." in green_flags and \
           "Confident tone detected." in green_flags:
            green_flags.append("Excellent combination: Positive sentiment, joy, and confident tone.")

        # Determine overall flag status
        overall_status = "Neutral"
        if red_flags and not green_flags:
            overall_status = "Red Flag"
        elif green_flags and not red_flags:
            overall_status = "Green Flag"
        elif red_flags and green_flags:
            overall_status = "Mixed (Both Red and Green Flags)"

        return {
            "overall_status": overall_status,
            "red_flags": list(set(red_flags)), # Use set to remove duplicates
            "green_flags": list(set(green_flags))
        }

if __name__ == '__main__':
    detector = FlagDetector()

    # Example 1: Clear Red Flag
    print("\n--- Example 1: Clear Red Flag ---")
    result1 = detector.detect_flags("Negative", "anger", "aggressive")
    print(result1)

    # Example 2: Clear Green Flag
    print("\n--- Example 2: Clear Green Flag ---")
    result2 = detector.detect_flags("Positive", "joy", "confident")
    print(result2)

    # Example 3: Neutral/Mixed
    print("\n--- Example 3: Neutral/Mixed ---")
    result3 = detector.detect_flags("Neutral", "neutral", "neutral")
    print(result3)

    # Example 4: Mixed scenario
    print("\n--- Example 4: Mixed Scenario ---")
    result4 = detector.detect_flags("Negative", "surprise", "tentative")
    print(result4)

    # Example 5: Another mixed scenario
    print("\n--- Example 5: Another Mixed Scenario ---")
    result5 = detector.detect_flags("Positive", "fear", "analytical")
    print(result5)


