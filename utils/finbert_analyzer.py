"""
FinBERT Sentiment Analyzer
Wraps HuggingFace ProsusAI/finbert model for financial text classification.
"""

import warnings
warnings.filterwarnings('ignore')

import os

class FinBERTAnalyzer:
    """Singleton for FinBERT Model to avoid reloading."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERTAnalyzer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def __init__(self):
        if self.initialized:
            return
            
        self.use_finbert = False
        self.pipeline = None
        
        print("System: Initializing FinBERT NLP model (this may take a moment if downloading)...")
        try:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            self.use_finbert = True
            print("System: FinBERT initialized successfully.")
        except ImportError:
            print("Warning: 'transformers' or 'torch' not installed. Falling back to TextBlob.")
        except Exception as e:
            print(f"Warning: FinBERT initialization failed: {e}. Falling back to TextBlob.")
            
        self.initialized = True

    def analyze(self, text: str) -> float:
        """
        Analyze text and return sentiment from -1.0 to 1.0.
        positive = > 0
        negative = < 0
        neutral = 0
        """
        if not text or len(text.strip()) == 0:
            return 0.0
            
        if self.use_finbert and self.pipeline:
            try:
                # FinBERT max length is 512 tokens. Safely truncate string.
                # 1 token is approx 4 chars
                safe_text = text[:2000]
                result = self.pipeline(safe_text)[0]
                
                label = result['label'] # positive, negative, neutral
                score = result['score'] # confidence 0 to 1
                
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:
                    return 0.0 # neutral
            except Exception as e:
                pass # fallback below
                
        # Fallback to TextBlob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

# Global instance
_analyzer = None

def get_finbert_sentiment(text: str) -> float:
    """Helper function to get sentiment using global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBERTAnalyzer()
    return _analyzer.analyze(text)

if __name__ == "__main__":
    tests = [
        "The Fed raised interest rates by 50 bps causing markets to tumble.",
        "Apple reports record breaking earnings for Q3.",
        "Bitcoin price remains stable as volume drops."
    ]
    for text in tests:
        score = get_finbert_sentiment(text)
        print(f"Score: {score:+.2f} | {text}")
