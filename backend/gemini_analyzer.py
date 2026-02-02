"""
Gemini LLM Analyzer Service
Uses Google Gemini for advanced sentiment analysis and market context
"""

import google.generativeai as genai
from typing import List, Dict, Optional
import json


class GeminiAnalyzer:
    """Use Gemini for advanced news analysis and market insights"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini analyzer
        
        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment of news articles using Gemini
        
        Args:
            articles: List of article dictionaries with titles and optional content
            
        Returns:
            Comprehensive sentiment analysis
        """
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0,
                'confidence': 0,
                'key_themes': [],
                'summary': 'No articles to analyze'
            }
        
        # Prepare articles for analysis
        articles_text = "\n\n".join([
            f"Source: {article.get('source', 'Unknown')}\n"
            f"Title: {article['title']}\n"
            f"Date: {article.get('published_date', 'Unknown')}"
            for article in articles[:10]  # Limit to avoid token limits
        ])
        
        prompt = f"""Analyze the sentiment and key themes from these news articles:

{articles_text}

Provide a JSON response with:
1. overall_sentiment: "very_positive", "positive", "neutral", "negative", or "very_negative"
2. sentiment_score: numeric value from -1 (very negative) to 1 (very positive)
3. confidence: confidence level from 0 to 1
4. key_themes: list of 3-5 main themes or topics
5. summary: 2-3 sentence summary of the news landscape
6. market_impact: potential impact on related markets ("bullish", "bearish", or "neutral")

Return only valid JSON, no additional text."""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            return result
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0,
                'confidence': 0,
                'key_themes': [],
                'summary': 'Error analyzing articles',
                'market_impact': 'neutral'
            }
    
    def analyze_market_context(self, market_title: str, market_description: str, 
                               news_summary: str) -> Dict:
        """
        Analyze market in context of current news
        
        Args:
            market_title: Kalshi market title
            market_description: Market description
            news_summary: Summary of related news
            
        Returns:
            Contextual analysis
        """
        prompt = f"""Analyze this prediction market in the context of current news:

Market: {market_title}
Description: {market_description}

Current News Context:
{news_summary}

Provide a JSON response with:
1. key_factors: list of 3-5 key factors that could influence this market
2. bullish_signals: list of positive indicators from the news
3. bearish_signals: list of negative indicators from the news
4. event_significance: how significant recent events are (scale 1-10)
5. recommendation_rationale: brief explanation of how news affects market prediction
6. risk_level: "low", "medium", or "high"

Return only valid JSON."""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            return result
        except Exception as e:
            print(f"Context analysis error: {e}")
            return {
                'key_factors': [],
                'bullish_signals': [],
                'bearish_signals': [],
                'event_significance': 5,
                'recommendation_rationale': 'Unable to analyze context',
                'risk_level': 'medium'
            }
    
    def generate_prediction_explanation(self, market_title: str, 
                                       prediction: str, 
                                       confidence: float,
                                       market_features: Dict,
                                       news_features: Dict) -> str:
        """
        Generate natural language explanation for prediction
        
        Args:
            market_title: Market title
            prediction: Prediction class (e.g., "Strong Buy")
            confidence: Confidence score
            market_features: Market data features
            news_features: News sentiment features
            
        Returns:
            Natural language explanation
        """
        prompt = f"""Generate a clear, concise explanation for this market prediction:

Market: {market_title}
Prediction: {prediction}
Confidence: {confidence:.1%}

Market Data:
- Current Price: ${market_features.get('current_price', 0):.2f}
- Volume (24h): {market_features.get('volume_24h', 0):,}
- Price Change (7d): {market_features.get('price_change_7d', 0):.2%}
- Volatility: {market_features.get('volatility', 0):.3f}

News Sentiment:
- Average Sentiment: {news_features.get('avg_sentiment', 0):.2f}
- Article Count: {news_features.get('article_count', 0)}
- Positive Ratio: {news_features.get('positive_ratio', 0):.1%}

Write a 3-4 sentence explanation that:
1. States the prediction clearly
2. Highlights the most important factors
3. Explains why the confidence is at this level
4. Mentions any key risks or caveats

Keep it conversational and easy to understand."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Explanation generation error: {e}")
            return f"Based on current market data and news sentiment, the prediction is {prediction} with {confidence:.1%} confidence."
    
    def extract_entities_and_events(self, articles: List[Dict]) -> Dict:
        """
        Extract key entities and events from news articles
        
        Args:
            articles: List of articles
            
        Returns:
            Entities and events analysis
        """
        articles_text = "\n\n".join([
            f"Title: {article['title']}"
            for article in articles[:15]
        ])
        
        prompt = f"""Analyze these news headlines and extract:

{articles_text}

Provide a JSON response with:
1. key_people: list of important people mentioned
2. key_organizations: list of organizations mentioned
3. key_events: list of significant events
4. trending_topics: list of recurring themes
5. controversy_score: how controversial the topic is (0-10)

Return only valid JSON."""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {
                'key_people': [],
                'key_organizations': [],
                'key_events': [],
                'trending_topics': [],
                'controversy_score': 5
            }
    
    def compare_sources(self, articles: List[Dict]) -> Dict:
        """
        Compare how different sources are covering the topic
        
        Args:
            articles: Articles from different sources
            
        Returns:
            Source comparison analysis
        """
        # Group by source
        by_source = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(article['title'])
        
        sources_text = "\n\n".join([
            f"{source}:\n" + "\n".join(f"- {title}" for title in titles)
            for source, titles in by_source.items()
        ])
        
        prompt = f"""Compare how different news sources are covering this topic:

{sources_text}

Provide a JSON response with:
1. consensus_topics: topics all sources agree on
2. divergent_coverage: how coverage differs between sources
3. bias_indicators: any signs of bias (if any)
4. reliability_score: overall reliability of the news landscape (0-10)

Return only valid JSON."""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
        except Exception as e:
            print(f"Source comparison error: {e}")
            return {
                'consensus_topics': [],
                'divergent_coverage': [],
                'bias_indicators': [],
                'reliability_score': 7
            }


# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
    else:
        analyzer = GeminiAnalyzer(api_key)
        
        # Example articles
        sample_articles = [
            {
                'source': 'CNN',
                'title': 'Trump announces new economic policy',
                'published_date': '2 hours ago'
            },
            {
                'source': 'Reuters',
                'title': 'Markets react to Trump statement',
                'published_date': '1 hour ago'
            }
        ]
        
        print("Analyzing sentiment...")
        sentiment = analyzer.analyze_sentiment(sample_articles)
        print(json.dumps(sentiment, indent=2))
        
        print("\n\nExtracting entities...")
        entities = analyzer.extract_entities_and_events(sample_articles)
        print(json.dumps(entities, indent=2))
