"""Sentiment module init."""
try:
    from .analyzer import SentimentAnalyzer, generate_simulated_news
except ImportError as e:
    import warnings
    warnings.warn(f"Sentiment module unavailable: {e}")
    SentimentAnalyzer = None
    generate_simulated_news = None
