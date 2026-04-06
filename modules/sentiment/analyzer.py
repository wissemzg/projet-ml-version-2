"""
Module 2 — Sentiment Analysis for BVMT stocks.
Supports French + Arabic multilingual analysis using GPT-4o via OpenAI API.
Includes news scraping simulation and daily sentiment aggregation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import re
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Multilingual sentiment analysis for Tunisian financial news using GPT-4o."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')

    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection for French/Arabic."""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]+')
        if arabic_pattern.search(text):
            return 'ar'
        return 'fr'

    def _call_gpt(self, text: str) -> dict:
        """Call GPT-4o to classify sentiment of a financial text."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        system_prompt = (
            "You are a financial sentiment analysis expert for the Tunisian stock market (BVMT). "
            "You understand French and Arabic. "
            "Classify the sentiment of the given financial text as exactly one of: positive, negative, neutral. "
            "Also provide a confidence score between 0 and 1, and a sentiment score between -1 (very negative) and 1 (very positive). "
            "Respond with ONLY a JSON object in this exact format, no extra text:\n"
            '{"sentiment": "positive|negative|neutral", "score": 0.75, "confidence": 0.9}'
        )
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:1000]}
            ],
            "temperature": 0.0,
            "max_tokens": 60
        }
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        # Parse the JSON response
        result = json.loads(content)
        return {
            'sentiment': result.get('sentiment', 'neutral'),
            'score': round(float(result.get('score', 0.0)), 4),
            'confidence': round(float(result.get('confidence', 0.5)), 4)
        }

    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of a single text using GPT-4o."""
        if not text or len(text.strip()) < 5:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0, 'language': 'unknown'}

        lang = self.detect_language(text)

        try:
            result = self._call_gpt(text)
            result['language'] = lang
            return result
        except Exception as e:
            print(f"GPT-4o sentiment fallback: {e}")
            return self._keyword_sentiment(text, lang)
    
    def _keyword_sentiment(self, text: str, lang: str) -> dict:
        """Keyword-based fallback sentiment analysis."""
        text_lower = text.lower()
        
        positive_fr = ['hausse', 'croissance', 'bénéfice', 'profit', 'progression',
                       'augmentation', 'record', 'positif', 'optimisme', 'dividende',
                       'reprise', 'succès', 'amélioration', 'performance']
        negative_fr = ['baisse', 'chute', 'perte', 'crise', 'déficit', 'recul',
                       'dégradation', 'risque', 'négatif', 'endettement', 'faillite',
                       'sanctions', 'inflation', 'dévaluation']
        positive_ar = ['ارتفاع', 'نمو', 'أرباح', 'تحسن', 'إيجابي', 'صعود']
        negative_ar = ['انخفاض', 'خسارة', 'أزمة', 'تراجع', 'سلبي', 'هبوط']
        
        pos_words = positive_fr + positive_ar
        neg_words = negative_fr + negative_ar
        
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        
        if pos_count > neg_count:
            return {'sentiment': 'positive', 'score': 0.6, 'confidence': 0.5, 'language': lang}
        elif neg_count > pos_count:
            return {'sentiment': 'negative', 'score': -0.6, 'confidence': 0.5, 'language': lang}
        return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.4, 'language': lang}
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze_text(t) for t in texts]
    
    def aggregate_daily_sentiment(self, articles: list) -> dict:
        """
        Compute daily aggregated sentiment from a list of articles.
        Each article: {'text': str, 'stock': str, 'date': str, 'source': str}
        """
        if not articles:
            return {"overall": "neutral", "score": 0.0, "count": 0, "by_stock": {}}
        
        results = []
        for art in articles:
            sent = self.analyze_text(art.get('text', ''))
            sent['stock'] = art.get('stock', 'UNKNOWN')
            sent['date'] = art.get('date', '')
            sent['source'] = art.get('source', '')
            results.append(sent)
        
        df = pd.DataFrame(results)
        
        # Overall
        avg_score = df['score'].mean()
        overall = 'positive' if avg_score > 0.15 else ('negative' if avg_score < -0.15 else 'neutral')
        
        # By stock
        by_stock = {}
        for stock, grp in df.groupby('stock'):
            stock_avg = grp['score'].mean()
            by_stock[stock] = {
                'sentiment': 'positive' if stock_avg > 0.15 else ('negative' if stock_avg < -0.15 else 'neutral'),
                'score': round(stock_avg, 4),
                'count': len(grp),
                'positive_pct': round((grp['sentiment'] == 'positive').mean() * 100, 1),
                'negative_pct': round((grp['sentiment'] == 'negative').mean() * 100, 1),
                'neutral_pct': round((grp['sentiment'] == 'neutral').mean() * 100, 1),
            }
        
        return {
            "overall": overall,
            "score": round(avg_score, 4),
            "count": len(articles),
            "by_stock": by_stock,
            "details": results
        }


def generate_simulated_news(stocks: list, num_days: int = 30) -> list:
    """
    Generate simulated Tunisian financial news for demo purposes.
    [ASSUMPTION] Real scraping would target: ilboursa.com, tustex.com, webmanagercenter.com
    """
    templates_fr = [
        ("{stock} enregistre une hausse de ses bénéfices au T4 2025", "positive"),
        ("Le titre {stock} recule face aux pressions inflationnistes", "negative"),
        ("{stock} annonce un dividende exceptionnel pour ses actionnaires", "positive"),
        ("Résultats mitigés pour {stock} au premier semestre", "neutral"),
        ("{stock} fait face à des difficultés de liquidité sur le marché", "negative"),
        ("Les analystes recommandent {stock} après les derniers résultats", "positive"),
        ("{stock} : la BVMT suspend temporairement la cotation", "negative"),
        ("Le volume d'échange de {stock} atteint un nouveau record", "positive"),
        ("{stock} maintient sa position malgré la volatilité du marché", "neutral"),
        ("Investissement stratégique de {stock} dans les nouvelles technologies", "positive"),
        ("{stock} : baisse des transactions suite aux incertitudes économiques", "negative"),
        ("Le CMF examine les pratiques commerciales de {stock}", "negative"),
        ("{stock} bénéficie de la reprise du secteur bancaire tunisien", "positive"),
    ]
    
    templates_ar = [
        ("سهم {stock} يرتفع بفضل نتائج مالية إيجابية", "positive"),
        ("انخفاض في قيمة سهم {stock} بسبب تراجع الأرباح", "negative"),
        ("{stock} يعلن عن توزيعات أرباح جديدة للمساهمين", "positive"),
        ("استقرار نسبي لسهم {stock} في بورصة تونس", "neutral"),
        ("تحذيرات من مخاطر الاستثمار في {stock}", "negative"),
    ]
    
    all_templates = templates_fr + templates_ar
    articles = []
    sources = ['ilboursa.com', 'tustex.com', 'webmanagercenter.com', 'kapitalis.com']
    
    base_date = datetime.now()
    
    for day in range(num_days):
        date = base_date - timedelta(days=day)
        date_str = date.strftime('%Y-%m-%d')
        
        # 2-5 articles per day
        n_articles = np.random.randint(2, 6)
        for _ in range(n_articles):
            stock = np.random.choice(stocks)
            template, expected_sent = all_templates[np.random.randint(len(all_templates))]
            text = template.format(stock=stock)
            
            articles.append({
                'text': text,
                'stock': stock,
                'date': date_str,
                'source': np.random.choice(sources),
                'expected_sentiment': expected_sent
            })
    
    return articles
