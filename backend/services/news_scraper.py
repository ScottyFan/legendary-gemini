"""
Enhanced News Scraper - REAL scraping with better features
"""

import requests
from typing import List, Dict
import re
from urllib.parse import quote
import random


class RobustNewsScraper:
    """News scraper with enhanced feature extraction"""
    
    def __init__(self, use_mock_fallback: bool = True):
        self.session = requests.Session()
        self.use_mock_fallback = use_mock_fallback
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        self._update_headers()
    
    def _update_headers(self):
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def search_news(self, query: str, sources: List[str] = None, max_articles: int = 20) -> List[Dict]:
        """Search for news"""
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip().lower()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        return self._try_real_scraping(clean_query, max_articles)
    
    def _try_real_scraping(self, query: str, max_articles: int) -> List[Dict]:
        """Scrape Google News RSS"""
        articles = []
        
        try:
            import feedparser
            rss_url = f'https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en'
            
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                for entry in feed.entries[:max_articles]:
                    try:
                        title = entry.get('title', '').strip()
                        source = 'Google News'
                        
                        if ' - ' in title:
                            parts = title.rsplit(' - ', 1)
                            title = parts[0]
                            source = parts[1]
                        
                        articles.append({
                            'source': source,
                            'title': title,
                            'url': entry.get('link', ''),
                            'published_date': entry.get('published', ''),
                            'query': query
                        })
                    except:
                        continue
        except Exception as e:
            pass
        
        return articles
    
    def calculate_basic_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment"""
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'success', 'win', 'winning',
            'strong', 'growth', 'rise', 'surge', 'rally', 'gain', 'profit',
            'boost', 'improve', 'optimistic', 'confidence', 'bullish', 'up',
            'high', 'increase', 'better', 'best', 'leading', 'ahead', 'surges',
            'soar', 'climb', 'record', 'breakthrough', 'triumph', 'victory'
        }
        
        negative_words = {
            'bad', 'terrible', 'negative', 'fail', 'failure', 'lose', 'losing',
            'weak', 'decline', 'fall', 'crash', 'drop', 'loss', 'deficit',
            'concern', 'worry', 'pessimistic', 'fear', 'bearish', 'crisis',
            'down', 'low', 'decrease', 'worse', 'worst', 'behind', 'risk',
            'challenges', 'concerns', 'trouble', 'struggle', 'plunge', 'sink'
        }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total = len(words)
        
        if total == 0:
            return {'score': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
        
        score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        
        return {'score': score}
    
    def extract_enhanced_news_features(self, queries: List[str]) -> Dict:
        """Extract ENHANCED news features from MULTIPLE queries"""
        
        all_articles = []
        
        # Search multiple queries
        for query in queries[:3]:
            articles = self.search_news(query, max_articles=10)
            all_articles.extend(articles)
        
        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # Defaults
        features = {
            'news_article_count': 0,
            'news_avg_sentiment': 0,
            'news_sentiment_std': 0,
            'news_positive_ratio': 0.5,
            'news_negative_ratio': 0.5,
            'news_source_diversity': 0,
            'news_recency_score': 0,
            'news_sentiment_range': 0,
            'news_max_positive': 0,
            'news_max_negative': 0,
            'news_controversy_score': 0,
            'news_volume_score': 0,
        }
        
        if not unique_articles:
            return features
        
        # Analyze
        sentiments = []
        sources = set()
        
        for article in unique_articles:
            sentiment = self.calculate_basic_sentiment(article['title'])
            sentiments.append(sentiment['score'])
            sources.add(article.get('source', 'Unknown'))
        
        # Calculate
        import statistics
        
        features['news_article_count'] = len(unique_articles)
        features['news_source_diversity'] = len(sources)
        
        if sentiments:
            features['news_avg_sentiment'] = statistics.mean(sentiments)
            features['news_sentiment_std'] = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
            features['news_positive_ratio'] = sum(1 for s in sentiments if s > 0.1) / len(sentiments)
            features['news_negative_ratio'] = sum(1 for s in sentiments if s < -0.1) / len(sentiments)
            
            features['news_sentiment_range'] = max(sentiments) - min(sentiments)
            features['news_max_positive'] = max(sentiments)
            features['news_max_negative'] = min(sentiments)
            features['news_controversy_score'] = features['news_sentiment_std'] * features['news_article_count']
            features['news_volume_score'] = min(features['news_article_count'] / 20, 1.0)
        
        # Recency
        recent_count = sum(1 for a in unique_articles 
                          if any(term in a.get('published_date', '').lower() 
                                for term in ['hour', 'minute', 'ago']))
        features['news_recency_score'] = recent_count / len(unique_articles)
        
        return features
    
    # BACKWARD COMPATIBILITY
    def extract_news_features(self, query: str, sources: List[str] = None) -> Dict:
        """Old method name - redirects to new enhanced version"""
        queries = [query] if isinstance(query, str) else query
        return self.extract_enhanced_news_features(queries)


# Alias
NewsScraper = RobustNewsScraper