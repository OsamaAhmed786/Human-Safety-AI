import re
from collections import Counter

class ToneAnalyzer:
    def __init__(self):
        # Define word lists for different tones
        
        # Aggressive/Negative Words
        self.aggressive_words = [
            "hate", "angry", "furious", "rage", "mad", "pissed", "annoyed", "irritated",
            "stupid", "idiot", "moron", "dumb", "pathetic", "worthless", "useless",
            "damn", "hell", "shit", "fuck", "asshole", "bitch", "bastard",
            "kill", "destroy", "crush", "attack", "fight", "war", "violence",
            "disgusting", "awful", "terrible", "horrible", "nasty", "gross",
            "outraged", "livid", "enraged", "bitter", "hostile", "aggressive",
            "revenge", "hatred", "despise", "loathe", "detest", "abhor",
            "toxic", "poisonous", "vicious", "cruel", "brutal", "savage",
            "ridiculous", "absurd", "nonsense", "garbage", "trash", "crap"
        ]
        
        # Confident/Assertive Words
        self.confident_words = [
            "confident", "sure", "certain", "definitely", "absolutely", "clearly",
            "obviously", "undoubtedly", "without doubt", "guaranteed", "proven",
            "strong", "powerful", "capable", "skilled", "expert", "professional",
            "determined", "decisive", "bold", "assertive", "fearless", "brave",
            "accomplished", "successful", "victorious", "triumphant", "dominant",
            "superior", "excellent", "outstanding", "exceptional", "remarkable",
            "unquestionably", "indisputable", "irrefutable", "conclusive", "final",
            "master", "champion", "winner", "leader", "authority", "specialist",
            "competent", "qualified", "experienced", "seasoned", "veteran", "ace",
            "flawless", "perfect", "impeccable", "stellar", "top-notch", "first-rate"
        ]
        
        # Tentative/Uncertain Words
        self.tentative_words = [
            "maybe", "perhaps", "possibly", "might", "could", "would", "should",
            "probably", "likely", "seems", "appears", "suggests", "indicates",
            "I think", "I believe", "I guess", "I suppose", "not sure", "uncertain",
            "doubtful", "hesitant", "unsure", "questionable", "debatable", "ambiguous",
            "potentially", "conceivably", "presumably", "allegedly", "supposedly", "reportedly",
            "tentatively", "provisionally", "hypothetically", "theoretically", "presumably",
            "roughly", "approximately", "around", "about", "nearly", "almost",
            "kind of", "sort of", "somewhat", "rather", "fairly", "quite",
            "I wonder", "it's possible", "hard to say", "difficult to tell", "unclear",
            "vague", "fuzzy", "blurry", "indefinite", "undecided", "on the fence"
        ]
        
        # Analytical/Logical Words
        self.analytical_words = [
            "analyze", "examine", "study", "research", "investigate", "evaluate",
            "assess", "consider", "compare", "contrast", "therefore", "however",
            "furthermore", "moreover", "consequently", "thus", "hence", "because",
            "systematic", "methodical", "logical", "rational", "objective", "empirical",
            "statistical", "quantitative", "qualitative", "experimental", "theoretical",
            "hypothesis", "conclusion", "evidence", "data", "findings", "results",
            "correlation", "causation", "pattern", "trend", "analysis", "synthesis",
            "deduction", "induction", "inference", "reasoning", "calculation", "measurement",
            "observation", "classification", "categorization", "interpretation", "validation",
            "verification", "substantiate", "demonstrate", "prove", "establish", "confirm"
        ]
        
        # Emotional/Positive Words
        self.emotional_words = [
            "love", "adore", "cherish", "treasure", "wonderful", "amazing",
            "fantastic", "incredible", "beautiful", "gorgeous", "stunning",
            "excited", "thrilled", "delighted", "overjoyed", "ecstatic",
            "passionate", "heartfelt", "touching", "moving", "inspiring", "uplifting",
            "joyful", "blissful", "euphoric", "elated", "cheerful", "happy",
            "grateful", "thankful", "appreciative", "blessed", "fortunate", "lucky",
            "magnificent", "spectacular", "breathtaking", "extraordinary", "phenomenal",
            "marvelous", "fabulous", "splendid", "glorious", "divine", "heavenly",
            "adorable", "charming", "delightful", "enchanting", "captivating", "mesmerizing",
            "warm", "cozy", "comfortable", "peaceful", "serene", "tranquil"
        ]
        
        # Formal/Professional Words
        self.formal_words = [
            "professional", "corporate", "official", "formal", "business", "executive",
            "pursuant", "heretofore", "aforementioned", "subsequent", "prior", "respective",
            "implementation", "optimization", "facilitation", "coordination", "administration",
            "regulation", "compliance", "protocol", "procedure", "policy", "guideline",
            "strategic", "tactical", "operational", "systematic", "comprehensive", "extensive",
            "substantial", "significant", "considerable", "notable", "prominent", "distinguished",
            "esteemed", "respected", "renowned", "established", "recognized", "certified",
            "authorized", "approved", "endorsed", "validated", "accredited", "licensed",
            "diplomatic", "courteous", "respectful", "appropriate", "suitable", "adequate",
            "satisfactory", "acceptable", "standard", "conventional", "traditional", "customary"
        ]
        
        # Casual/Informal Words
        self.casual_words = [
            "yeah", "yep", "nope", "cool", "awesome", "sweet", "nice", "dude",
            "buddy", "pal", "mate", "bro", "guy", "folks", "guys", "kidding",
            "joking", "funny", "hilarious", "crazy", "wild", "insane", "nuts",
            "totally", "absolutely", "definitely", "seriously", "literally", "basically",
            "actually", "really", "pretty", "super", "ultra", "mega", "hyper",
            "chill", "relax", "hang out", "hang around", "mess around", "fool around",
            "stuff", "things", "whatever", "anyways", "anyhow", "somehow", "somewhere",
            "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
            "c'mon", "come on", "no way", "way to go", "right on", "far out"
        ]
        
        # Urgent/Action Words
        self.urgent_words = [
            "urgent", "emergency", "immediate", "critical", "crucial", "vital",
            "essential", "important", "priority", "deadline", "rush", "hurry",
            "quickly", "rapidly", "instantly", "immediately", "promptly", "swiftly",
            "action", "act", "do", "execute", "implement", "perform", "achieve",
            "accomplish", "complete", "finish", "solve", "fix", "resolve", "address",
            "handle", "manage", "deal with", "take care of", "attend to", "focus on",
            "concentrate", "emphasize", "stress", "highlight", "underline", "prioritize",
            "expedite", "accelerate", "speed up", "fast-track", "push forward", "advance",
            "now", "today", "asap", "right away", "at once", "without delay"
        ]
        
        # Questioning/Inquiry Words
        self.questioning_words = [
            "what", "when", "where", "why", "how", "who", "which", "whose",
            "question", "ask", "inquire", "wonder", "curious", "investigate",
            "explore", "discover", "find out", "learn", "understand", "clarify",
            "explain", "elaborate", "specify", "detail", "describe", "define",
            "puzzle", "mystery", "riddle", "enigma", "confusion", "bewilderment",
            "doubt", "skeptical", "suspicious", "questionable", "dubious", "uncertain",
            "query", "interrogate", "probe", "examine", "inspect", "scrutinize",
            "challenge", "contest", "dispute", "debate", "argue", "discuss",
            "confirm", "verify", "validate", "check", "test", "trial", "experiment"
        ]
        
        # Supportive/Encouraging Words
        self.supportive_words = [
            "support", "encourage", "motivate", "inspire", "boost", "uplift",
            "help", "assist", "aid", "guide", "mentor", "coach", "teach",
            "believe", "trust", "faith", "hope", "optimism", "positive", "bright",
            "promising", "potential", "capable", "talented", "gifted", "skilled",
            "congratulations", "well done", "good job", "excellent work", "bravo",
            "proud", "admire", "respect", "appreciate", "value", "honor", "praise",
            "comfort", "console", "reassure", "calm", "soothe", "peaceful", "gentle",
            "kind", "caring", "compassionate", "understanding", "empathetic", "sympathetic",
            "together", "unity", "cooperation", "collaboration", "teamwork", "partnership"
        ]
        
        # Enhanced punctuation patterns
        self.exclamation_pattern = r'!+'
        self.question_pattern = r'\?+'
        self.caps_pattern = r'\b[A-Z]{2,}\b'
        self.ellipsis_pattern = r'\.{3,}'
        
    def count_word_matches(self, text, word_list):
        """Count how many words from word_list appear in the text with improved matching."""
        text_lower = text.lower()
        # Remove punctuation for better matching
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words_in_text = text_clean.split()
        
        count = 0
        matched_words = []
        
        for word in word_list:
            word_lower = word.lower()
            # Check for exact word match and phrase match
            if word_lower in words_in_text or word_lower in text_lower:
                count += 1
                matched_words.append(word)
        
        return count, matched_words
    
    def analyze_punctuation(self, text):
        """Analyze punctuation patterns in the text with enhanced detection."""
        exclamations = len(re.findall(self.exclamation_pattern, text))
        questions = len(re.findall(self.question_pattern, text))
        caps_words = len(re.findall(self.caps_pattern, text))
        ellipsis = len(re.findall(self.ellipsis_pattern, text))
        
        # Additional punctuation analysis
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))  # Like "sooo" or "yesss"
        
        return {
            'exclamations': exclamations,
            'questions': questions,
            'caps_words': caps_words,
            'ellipsis': ellipsis,
            'repeated_chars': repeated_chars
        }
    
    def get_text_statistics(self, text):
        """Get comprehensive text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word.strip('.,!?;:')) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def analyze_tone(self, text):
        """
        Enhanced tone analysis with all categories and improved scoring.
        Returns a comprehensive dictionary with tone scores and analysis.
        """
        if not text or not isinstance(text, str):
            return self._get_empty_analysis()
        
        # Get text statistics
        text_stats = self.get_text_statistics(text)
        
        # Count matches for each tone category (now returns matched words too)
        aggressive_score, aggressive_matches = self.count_word_matches(text, self.aggressive_words)
        confident_score, confident_matches = self.count_word_matches(text, self.confident_words)
        tentative_score, tentative_matches = self.count_word_matches(text, self.tentative_words)
        analytical_score, analytical_matches = self.count_word_matches(text, self.analytical_words)
        emotional_score, emotional_matches = self.count_word_matches(text, self.emotional_words)
        formal_score, formal_matches = self.count_word_matches(text, self.formal_words)
        casual_score, casual_matches = self.count_word_matches(text, self.casual_words)
        urgent_score, urgent_matches = self.count_word_matches(text, self.urgent_words)
        questioning_score, questioning_matches = self.count_word_matches(text, self.questioning_words)
        supportive_score, supportive_matches = self.count_word_matches(text, self.supportive_words)
        
        # Analyze punctuation
        punct_analysis = self.analyze_punctuation(text)
        
        # Enhanced punctuation-based adjustments
        if punct_analysis['exclamations'] > 0:
            emotional_score += punct_analysis['exclamations'] * 1.5
            aggressive_score += punct_analysis['exclamations'] * 1.0
            urgent_score += punct_analysis['exclamations'] * 0.8
            supportive_score += punct_analysis['exclamations'] * 0.5
        
        if punct_analysis['caps_words'] > 0:
            aggressive_score += punct_analysis['caps_words'] * 1.0
            confident_score += punct_analysis['caps_words'] * 0.7
            urgent_score += punct_analysis['caps_words'] * 0.8
        
        if punct_analysis['questions'] > 0:
            tentative_score += punct_analysis['questions'] * 0.8
            questioning_score += punct_analysis['questions'] * 1.2
            analytical_score += punct_analysis['questions'] * 0.4
        
        if punct_analysis['ellipsis'] > 0:
            tentative_score += punct_analysis['ellipsis'] * 0.6
            emotional_score += punct_analysis['ellipsis'] * 0.3
        
        if punct_analysis['repeated_chars'] > 0:
            casual_score += punct_analysis['repeated_chars'] * 0.5
            emotional_score += punct_analysis['repeated_chars'] * 0.4
        
        # Context-based adjustments
        word_count = text_stats['word_count']
        
        # Longer texts tend to be more analytical or formal
        if word_count > 100:
            analytical_score += 1.0
            formal_score += 0.7
        elif word_count > 50:
            analytical_score += 0.5
            formal_score += 0.3
        
        # Very short texts with casual words are likely informal
        if word_count < 15 and casual_score > 0:
            casual_score += 1.0
        
        # Compile all scores
        tone_scores = {
            'aggressive': aggressive_score,
            'confident': confident_score,
            'tentative': tentative_score,
            'analytical': analytical_score,
            'emotional': emotional_score,
            'formal': formal_score,
            'casual': casual_score,
            'urgent': urgent_score,
            'questioning': questioning_score,
            'supportive': supportive_score
        }
        
        # Find dominant tone
        dominant_tone = max(tone_scores, key=tone_scores.get)
        if tone_scores[dominant_tone] == 0:
            dominant_tone = 'neutral'
        
        # Calculate confidence level for dominant tone
        total_score = sum(tone_scores.values())
        confidence = (tone_scores[dominant_tone] / total_score * 100) if total_score > 0 else 0
        
        # Normalize scores to percentages
        normalized_scores = {}
        if total_score > 0:
            normalized_scores = {k: round((v / total_score) * 100, 2) for k, v in tone_scores.items()}
        else:
            normalized_scores = {k: 0 for k in tone_scores.keys()}
        
        # Store matched words for transparency
        matched_words = {
            'aggressive': aggressive_matches[:5],  # Show top 5 matches
            'confident': confident_matches[:5],
            'tentative': tentative_matches[:5],
            'analytical': analytical_matches[:5],
            'emotional': emotional_matches[:5],
            'formal': formal_matches[:5],
            'casual': casual_matches[:5],
            'urgent': urgent_matches[:5],
            'questioning': questioning_matches[:5],
            'supportive': supportive_matches[:5]
        }
        
        return {
            'raw_scores': tone_scores,
            'normalized_scores': normalized_scores,
            'dominant_tone': dominant_tone,
            'confidence_level': round(confidence, 2),
            'punctuation_analysis': punct_analysis,
            'text_statistics': text_stats,
            'matched_words': matched_words,
            'analysis_summary': self._generate_summary(dominant_tone, confidence, tone_scores)
        }
    
    def _get_empty_analysis(self):
        """Return empty analysis for invalid input."""
        empty_scores = {
            'aggressive': 0, 'confident': 0, 'tentative': 0, 'analytical': 0,
            'emotional': 0, 'formal': 0, 'casual': 0, 'urgent': 0,
            'questioning': 0, 'supportive': 0
        }
        
        return {
            'raw_scores': empty_scores,
            'normalized_scores': empty_scores,
            'dominant_tone': 'neutral',
            'confidence_level': 0,
            'punctuation_analysis': {'exclamations': 0, 'questions': 0, 'caps_words': 0, 'ellipsis': 0, 'repeated_chars': 0},
            'text_statistics': {'word_count': 0, 'char_count': 0, 'sentence_count': 0, 'avg_word_length': 0, 'avg_sentence_length': 0},
            'matched_words': {k: [] for k in empty_scores.keys()},
            'analysis_summary': 'No valid input provided.'
        }
    
    def _generate_summary(self, dominant_tone, confidence, tone_scores):
        """Generate a human-readable summary of the analysis."""
        summary = f"Primary tone: {dominant_tone.title()} ({confidence:.1f}% confidence)"
        
        # Find secondary tones
        sorted_tones = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_tones = [tone for tone, score in sorted_tones[1:3] if score > 0]
        
        if secondary_tones:
            summary += f" | Secondary tones: {', '.join([t.title() for t in secondary_tones])}"
        
        return summary
    
    def get_detailed_report(self, text):
        """Generate a comprehensive, human-readable report."""
        analysis = self.analyze_tone(text)
        
        report = f"""
=== TONE ANALYSIS REPORT ===

Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

{analysis['analysis_summary']}

--- DETAILED BREAKDOWN ---
"""
        
        # Show top 3 tones with their scores
        top_tones = sorted(analysis['normalized_scores'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        
        for tone, score in top_tones:
            if score > 0:
                report += f"\n{tone.upper()}: {score}%"
                if analysis['matched_words'][tone]:
                    report += f" (Triggered by: {', '.join(analysis['matched_words'][tone])})"
        
        # Text statistics
        stats = analysis['text_statistics']
        report += f"""

--- TEXT STATISTICS ---
Words: {stats['word_count']} | Characters: {stats['char_count']} | Sentences: {stats['sentence_count']}
Avg word length: {stats['avg_word_length']:.1f} | Avg sentence length: {stats['avg_sentence_length']:.1f}
"""
        
        # Punctuation analysis
        punct = analysis['punctuation_analysis']
        if any(punct.values()):
            report += "\n--- PUNCTUATION PATTERNS ---"
            if punct['exclamations']: report += f"\nExclamation marks: {punct['exclamations']}"
            if punct['questions']: report += f"\nQuestion marks: {punct['questions']}"
            if punct['caps_words']: report += f"\nALL CAPS words: {punct['caps_words']}"
            if punct['ellipsis']: report += f"\nEllipsis: {punct['ellipsis']}"
            if punct['repeated_chars']: report += f"\nRepeated characters: {punct['repeated_chars']}"
        
        return report

# Example usage and testing
if __name__ == '__main__':
    analyzer = ToneAnalyzer()

    # Test cases with various tones
    test_texts = [
        "I am absolutely confident that this is the right approach!",
        "I think maybe we should consider this option, but I'm not sure...",
        "This is STUPID and I HATE it! What the hell were you thinking?!",
        "Let's analyze the data carefully and examine the results systematically.",
        "I LOVE this so much! It's absolutely AMAZING and wonderful!!!",
        "The weather is nice today.",
        "Hey dude, that's totally awesome! Let's hang out later, yeah?",
        "URGENT: We need to complete this ASAP! This is critical!",
        "What do you think about this? I'm curious to know your opinion.",
        "You did an excellent job! I'm so proud of your achievements. Keep it up!"
    ]

    print("=== TONE ANALYZER TEST RESULTS ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"--- Test {i} ---")
        analysis = analyzer.analyze_tone(text)
        print(f"Text: \"{text}\"")
        print(f"Analysis: {analysis['analysis_summary']}")
        print(f"Top 3 Tones: {dict(sorted(analysis['normalized_scores'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        print()
    
    # Detailed report example
    print("\n=== DETAILED REPORT EXAMPLE ===")
    sample_text = "I'm absolutely FURIOUS! This is completely unacceptable and I HATE how this was handled!!!"
    print(analyzer.get_detailed_report(sample_text))



































# import re
# from collections import Counter

# class ToneAnalyzer:
#     def __init__(self):
#         # Define word lists for different tones
#        # Aggressive/Negative Words
#         self.aggressive_words = [
#             "hate", "angry", "furious", "rage", "mad", "pissed", "annoyed", "irritated",
#             "stupid", "idiot", "moron", "dumb", "pathetic", "worthless", "useless",
#             "damn", "hell", "shit", "fuck", "asshole", "bitch", "bastard",
#             "kill", "destroy", "crush", "attack", "fight", "war", "violence",
#             "disgusting", "awful", "terrible", "horrible", "nasty", "gross",
#             "outraged", "livid", "enraged", "bitter", "hostile", "aggressive",
#             "revenge", "hatred", "despise", "loathe", "detest", "abhor",
#             "toxic", "poisonous", "vicious", "cruel", "brutal", "savage",
#             "ridiculous", "absurd", "nonsense", "garbage", "trash", "crap"
#         ]
        
#         # Confident/Assertive Words
#         self.confident_words = [
#             "confident", "sure", "certain", "definitely", "absolutely", "clearly",
#             "obviously", "undoubtedly", "without doubt", "guaranteed", "proven",
#             "strong", "powerful", "capable", "skilled", "expert", "professional",
#             "determined", "decisive", "bold", "assertive", "fearless", "brave",
#             "accomplished", "successful", "victorious", "triumphant", "dominant",
#             "superior", "excellent", "outstanding", "exceptional", "remarkable",
#             "unquestionably", "indisputable", "irrefutable", "conclusive", "final",
#             "master", "champion", "winner", "leader", "authority", "specialist",
#             "competent", "qualified", "experienced", "seasoned", "veteran", "ace",
#             "flawless", "perfect", "impeccable", "stellar", "top-notch", "first-rate"
#         ]
        
#         # Tentative/Uncertain Words
#         self.tentative_words = [
#             "maybe", "perhaps", "possibly", "might", "could", "would", "should",
#             "probably", "likely", "seems", "appears", "suggests", "indicates",
#             "I think", "I believe", "I guess", "I suppose", "not sure", "uncertain",
#             "doubtful", "hesitant", "unsure", "questionable", "debatable", "ambiguous",
#             "potentially", "conceivably", "presumably", "allegedly", "supposedly", "reportedly",
#             "tentatively", "provisionally", "hypothetically", "theoretically", "presumably",
#             "roughly", "approximately", "around", "about", "nearly", "almost",
#             "kind of", "sort of", "somewhat", "rather", "fairly", "quite",
#             "I wonder", "it's possible", "hard to say", "difficult to tell", "unclear",
#             "vague", "fuzzy", "blurry", "indefinite", "undecided", "on the fence"
#         ]
        
#         # Analytical/Logical Words
#         self.analytical_words = [
#             "analyze", "examine", "study", "research", "investigate", "evaluate",
#             "assess", "consider", "compare", "contrast", "therefore", "however",
#             "furthermore", "moreover", "consequently", "thus", "hence", "because",
#             "systematic", "methodical", "logical", "rational", "objective", "empirical",
#             "statistical", "quantitative", "qualitative", "experimental", "theoretical",
#             "hypothesis", "conclusion", "evidence", "data", "findings", "results",
#             "correlation", "causation", "pattern", "trend", "analysis", "synthesis",
#             "deduction", "induction", "inference", "reasoning", "calculation", "measurement",
#             "observation", "classification", "categorization", "interpretation", "validation",
#             "verification", "substantiate", "demonstrate", "prove", "establish", "confirm"
#         ]
        
#         # Emotional/Positive Words
#         self.emotional_words = [
#             "love", "adore", "cherish", "treasure", "wonderful", "amazing",
#             "fantastic", "incredible", "beautiful", "gorgeous", "stunning",
#             "excited", "thrilled", "delighted", "overjoyed", "ecstatic",
#             "passionate", "heartfelt", "touching", "moving", "inspiring", "uplifting",
#             "joyful", "blissful", "euphoric", "elated", "cheerful", "happy",
#             "grateful", "thankful", "appreciative", "blessed", "fortunate", "lucky",
#             "magnificent", "spectacular", "breathtaking", "extraordinary", "phenomenal",
#             "marvelous", "fabulous", "splendid", "glorious", "divine", "heavenly",
#             "adorable", "charming", "delightful", "enchanting", "captivating", "mesmerizing",
#             "warm", "cozy", "comfortable", "peaceful", "serene", "tranquil"
#         ]
        
#         # Formal/Professional Words
#         self.formal_words = [
#             "professional", "corporate", "official", "formal", "business", "executive",
#             "pursuant", "heretofore", "aforementioned", "subsequent", "prior", "respective",
#             "implementation", "optimization", "facilitation", "coordination", "administration",
#             "regulation", "compliance", "protocol", "procedure", "policy", "guideline",
#             "strategic", "tactical", "operational", "systematic", "comprehensive", "extensive",
#             "substantial", "significant", "considerable", "notable", "prominent", "distinguished",
#             "esteemed", "respected", "renowned", "established", "recognized", "certified",
#             "authorized", "approved", "endorsed", "validated", "accredited", "licensed",
#             "diplomatic", "courteous", "respectful", "appropriate", "suitable", "adequate",
#             "satisfactory", "acceptable", "standard", "conventional", "traditional", "customary"
#         ]
        
#         # Casual/Informal Words
#         self.casual_words = [
#             "yeah", "yep", "nope", "cool", "awesome", "sweet", "nice", "dude",
#             "buddy", "pal", "mate", "bro", "guy", "folks", "guys", "kidding",
#             "joking", "funny", "hilarious", "crazy", "wild", "insane", "nuts",
#             "totally", "absolutely", "definitely", "seriously", "literally", "basically",
#             "actually", "really", "pretty", "super", "ultra", "mega", "hyper",
#             "chill", "relax", "hang out", "hang around", "mess around", "fool around",
#             "stuff", "things", "whatever", "anyways", "anyhow", "somehow", "somewhere",
#             "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
#             "c'mon", "come on", "no way", "way to go", "right on", "far out"
#         ]
        
#         # Urgent/Action Words
#         self.urgent_words = [
#             "urgent", "emergency", "immediate", "critical", "crucial", "vital",
#             "essential", "important", "priority", "deadline", "rush", "hurry",
#             "quickly", "rapidly", "instantly", "immediately", "promptly", "swiftly",
#             "action", "act", "do", "execute", "implement", "perform", "achieve",
#             "accomplish", "complete", "finish", "solve", "fix", "resolve", "address",
#             "handle", "manage", "deal with", "take care of", "attend to", "focus on",
#             "concentrate", "emphasize", "stress", "highlight", "underline", "prioritize",
#             "expedite", "accelerate", "speed up", "fast-track", "push forward", "advance",
#             "now", "today", "asap", "right away", "at once", "without delay"
#         ]
        
#         # Questioning/Inquiry Words
#         self.questioning_words = [
#             "what", "when", "where", "why", "how", "who", "which", "whose",
#             "question", "ask", "inquire", "wonder", "curious", "investigate",
#             "explore", "discover", "find out", "learn", "understand", "clarify",
#             "explain", "elaborate", "specify", "detail", "describe", "define",
#             "puzzle", "mystery", "riddle", "enigma", "confusion", "bewilderment",
#             "doubt", "skeptical", "suspicious", "questionable", "dubious", "uncertain",
#             "query", "interrogate", "probe", "examine", "inspect", "scrutinize",
#             "challenge", "contest", "dispute", "debate", "argue", "discuss",
#             "confirm", "verify", "validate", "check", "test", "trial", "experiment"
#         ]
        
#         # Supportive/Encouraging Words
#         self.supportive_words = [
#             "support", "encourage", "motivate", "inspire", "boost", "uplift",
#             "help", "assist", "aid", "guide", "mentor", "coach", "teach",
#             "believe", "trust", "faith", "hope", "optimism", "positive", "bright",
#             "promising", "potential", "capable", "talented", "gifted", "skilled",
#             "congratulations", "well done", "good job", "excellent work", "bravo",
#             "proud", "admire", "respect", "appreciate", "value", "honor", "praise",
#             "comfort", "console", "reassure", "calm", "soothe", "peaceful", "gentle",
#             "kind", "caring", "compassionate", "understanding", "empathetic", "sympathetic",
#             "together", "unity", "cooperation", "collaboration", "teamwork", "partnership"
#         ]
        
#         # Punctuation patterns
#         self.exclamation_pattern = r'!+'
#         self.question_pattern = r'\?+'
#         self.caps_pattern = r'[A-Z]{2,}'
        
#     def count_word_matches(self, text, word_list):
#         """Count how many words from word_list appear in the text."""
#         text_lower = text.lower()
#         count = 0
#         for word in word_list:
#             if word in text_lower:
#                 count += 1
#         return count
    
#     def analyze_punctuation(self, text):
#         """Analyze punctuation patterns in the text."""
#         exclamations = len(re.findall(self.exclamation_pattern, text))
#         questions = len(re.findall(self.question_pattern, text))
#         caps_words = len(re.findall(self.caps_pattern, text))
        
#         return {
#             'exclamations': exclamations,
#             'questions': questions,
#             'caps_words': caps_words
#         }
    
#     def analyze_tone(self, text):
#         """
#         Analyzes the tone of the given text using heuristic-based approaches.
#         Returns a dictionary with tone scores and the dominant tone.
#         """
#         if not text or not isinstance(text, str):
#             return {
#                 'aggressive': 0,
#                 'confident': 0,
#                 'tentative': 0,
#                 'analytical': 0,
#                 'emotional': 0,
#                 'dominant_tone': 'neutral',
#                 'punctuation_analysis': {'exclamations': 0, 'questions': 0, 'caps_words': 0}
#             }
        
#         # Count matches for each tone category
#         aggressive_score = self.count_word_matches(text, self.aggressive_words)
#         confident_score = self.count_word_matches(text, self.confident_words)
#         tentative_score = self.count_word_matches(text, self.tentative_words)
#         analytical_score = self.count_word_matches(text, self.analytical_words)
#         emotional_score = self.count_word_matches(text, self.emotional_words)
        
#         # Analyze punctuation
#         punct_analysis = self.analyze_punctuation(text)
        
#         # Adjust scores based on punctuation
#         if punct_analysis['exclamations'] > 0:
#             emotional_score += punct_analysis['exclamations']
#             aggressive_score += punct_analysis['exclamations'] * 0.5
        
#         if punct_analysis['caps_words'] > 0:
#             aggressive_score += punct_analysis['caps_words'] * 0.5
#             confident_score += punct_analysis['caps_words'] * 0.3
        
#         if punct_analysis['questions'] > 0:
#             tentative_score += punct_analysis['questions'] * 0.5
        
#         # Determine dominant tone
#         tone_scores = {
#             'aggressive': aggressive_score,
#             'confident': confident_score,
#             'tentative': tentative_score,
#             'analytical': analytical_score,
#             'emotional': emotional_score
#         }
        
#         dominant_tone = max(tone_scores, key=tone_scores.get)
#         if tone_scores[dominant_tone] == 0:
#             dominant_tone = 'neutral'
        
#         return {
#             'aggressive': aggressive_score,
#             'confident': confident_score,
#             'tentative': tentative_score,
#             'analytical': analytical_score,
#             'emotional': emotional_score,
#             'dominant_tone': dominant_tone,
#             'punctuation_analysis': punct_analysis
#         }

# if __name__ == '__main__':
#     analyzer = ToneAnalyzer()

#     texts = [
#         "I am absolutely confident that this is the right approach!",
#         "I think maybe we should consider this option, but I'm not sure.",
#         "This is STUPID and I HATE it! What the hell were you thinking?!",
#         "Let's analyze the data carefully and examine the results systematically.",
#         "I LOVE this so much! It's absolutely AMAZING and wonderful!!!",
#         "The weather is nice today."
#     ]

#     for text in texts:
#         tone_analysis = analyzer.analyze_tone(text)
#         print(f"Text: \"{text}\"")
#         print(f"Tone Analysis: {tone_analysis}")
#         print(f"Dominant Tone: {tone_analysis['dominant_tone']}\n")

