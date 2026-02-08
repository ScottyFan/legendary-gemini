"""
Enhanced News Scraper with Mock Data Fallback
Handles network restrictions gracefully
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, quote_plus, quote
import time
import random


class RobustNewsScraper:
    """
    News scraper with multiple fallback strategies:
    1. Try Google News RSS
    2. Try direct web scraping with retry
    3. Fall back to mock data if network restricted
    """
    
    def __init__(self, use_mock_fallback: bool = True):
        """
        Initialize scraper
        
        Args:
            use_mock_fallback: Whether to use mock data if real scraping fails
        """
        self.session = requests.Session()
        self.use_mock_fallback = use_mock_fallback
        
        # Realistic User-Agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        self._update_headers()
        
        # Mock news database for fallback
        self.mock_news_db = self._create_mock_news_database()
    
    def _update_headers(self):
        """Update session headers"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _create_mock_news_database(self) -> Dict[str, List[Dict]]:
        """Create realistic mock news for testing"""
        return {
            'trump': [
                {
                    'source': 'Reuters',
                    'title': 'Trump announces new economic policy ahead of 2026 midterms',
                    'url': 'https://reuters.com/article/trump-economy',
                    'published_date': '2 hours ago',
                    'sentiment_hint': 'neutral'
                },
                {
                    'source': 'CNN',
                    'title': 'Former president Trump rallies supporters in key swing state',
                    'url': 'https://cnn.com/politics/trump-rally',
                    'published_date': '5 hours ago',
                    'sentiment_hint': 'neutral'
                },
                {
                    'source': 'BBC',
                    'title': 'Trump legal challenges continue as prosecutors seek testimony',
                    'url': 'https://bbc.com/news/trump-legal',
                    'published_date': '1 day ago',
                    'sentiment_hint': 'negative'
                },
                {
                    'source': 'AP News',
                    'title': 'Markets react to Trump policy announcements',
                    'url': 'https://apnews.com/trump-markets',
                    'published_date': '3 hours ago',
                    'sentiment_hint': 'positive'
                },
                {
                    'source': 'Wall Street Journal',
                    'title': 'Business leaders respond to Trump economic proposals',
                    'url': 'https://wsj.com/politics/trump-business',
                    'published_date': '6 hours ago',
                    'sentiment_hint': 'neutral'
                }
            ],
            'bitcoin': [
                {
                    'source': 'CoinDesk',
                    'title': 'Bitcoin surges to new highs amid institutional buying',
                    'url': 'https://coindesk.com/bitcoin-surge',
                    'published_date': '1 hour ago',
                    'sentiment_hint': 'positive'
                },
                {
                    'source': 'Bloomberg',
                    'title': 'Cryptocurrency market shows strong momentum in early 2026',
                    'url': 'https://bloomberg.com/crypto-momentum',
                    'published_date': '3 hours ago',
                    'sentiment_hint': 'positive'
                },
                {
                    'source': 'Reuters',
                    'title': 'Bitcoin volatility concerns regulators as prices fluctuate',
                    'url': 'https://reuters.com/bitcoin-volatility',
                    'published_date': '5 hours ago',
                    'sentiment_hint': 'negative'
                }
            ],
            'election': [
                {
                    'source': 'New York Times',
                    'title': '2026 midterm elections: Key races to watch',
                    'url': 'https://nytimes.com/politics/midterms',
                    'published_date': '4 hours ago',
                    'sentiment_hint': 'neutral'
                },
                {
                    'source': 'Politico',
                    'title': 'Polling shows tight races in Senate battleground states',
                    'url': 'https://politico.com/senate-polling',
                    'published_date': '2 hours ago',
                    'sentiment_hint': 'neutral'
                }
            ],
            'sports': [
                {
                    'source': 'ESPN',
                    'title': 'NBA playoffs: Top teams battle for championship spots',
                    'url': 'https://espn.com/nba/playoffs',
                    'published_date': '1 hour ago',
                    'sentiment_hint': 'positive'
                },
                {
                    'source': 'Sports Illustrated',
                    'title': 'Star player injury concerns team ahead of crucial game',
                    'url': 'https://si.com/injury-concerns',
                    'published_date': '3 hours ago',
                    'sentiment_hint': 'negative'
                }
            ]
        }
    
    def search_news(self, query: str, sources: List[str] = None, max_articles: int = 20) -> List[Dict]:
        """
        Search for news with fallback to mock data
        
        Args:
            query: Search query
            sources: Ignored in this version
            max_articles: Maximum articles to return
            
        Returns:
            List of articles
        """
        # Clean query
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip().lower()
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        # Try real scraping first
        articles = self._try_real_scraping(clean_query, max_articles)
        
        # If failed and mock fallback enabled, use mock data
        if not articles and self.use_mock_fallback:
            print(f"  → Using mock news data for '{query}' (network restricted)")
            articles = self._get_mock_articles(clean_query, max_articles)
        
        return articles
    
    def _try_real_scraping(self, query: str, max_articles: int) -> List[Dict]:
        """Attempt real news scraping"""
        articles = []
        
        # Try Google News RSS (simple HTTP, less likely to be blocked)
        try:
            import feedparser
            rss_url = f'https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en'
            
            # Simple request without session complexity
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                for entry in feed.entries[:max_articles]:
                    try:
                        articles.append({
                            'source': 'Google News',
                            'title': entry.get('title', '').strip(),
                            'url': entry.get('link', ''),
                            'published_date': entry.get('published', ''),
                            'query': query
                        })
                    except:
                        continue
                
                if articles:
                    print(f"  ✓ Got {len(articles)} articles from Google News RSS")
                    return articles
        except Exception as e:
            pass
        
        return []
    
    def _get_mock_articles(self, query: str, max_articles: int) -> List[Dict]:
        """Get mock articles based on query keywords"""
        articles = []
        
        # Match query to mock categories
        query_lower = query.lower()
        
        # Direct matches
        if 'trump' in query_lower:
            articles = self.mock_news_db['trump']
        elif 'bitcoin' in query_lower or 'crypto' in query_lower:
            articles = self.mock_news_db['bitcoin']
        elif 'election' in query_lower or 'vote' in query_lower or 'senate' in query_lower:
            articles = self.mock_news_db['election']
        elif any(sport in query_lower for sport in ['nba', 'nfl', 'sport', 'game', 'player']):
            articles = self.mock_news_db['sports']
        else:
            # Generic financial news for markets
            articles = [
                {
                    'source': 'Financial Times',
                    'title': f'Market analysis: {query.title()} shows mixed signals',
                    'url': f'https://ft.com/markets/{query.replace(" ", "-")}',
                    'published_date': '2 hours ago',
                    'sentiment_hint': 'neutral'
                },
                {
                    'source': 'Bloomberg',
                    'title': f'Investors watch {query.title()} developments closely',
                    'url': f'https://bloomberg.com/{query.replace(" ", "-")}',
                    'published_date': '4 hours ago',
                    'sentiment_hint': 'neutral'
                },
                {
                    'source': 'Reuters',
                    'title': f'{query.title()} update: Key factors to consider',
                    'url': f'https://reuters.com/article/{query.replace(" ", "-")}',
                    'published_date': '6 hours ago',
                    'sentiment_hint': 'neutral'
                }
            ]
        
        # Add query to each article
        for article in articles:
            article['query'] = query
        
        return articles[:max_articles]
    
    def calculate_basic_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment from text"""
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'success', 'win', 'winning',
            'strong', 'growth', 'rise', 'surge', 'rally', 'gain', 'profit',
            'boost', 'improve', 'optimistic', 'confidence', 'bullish', 'up',
            'high', 'increase', 'better', 'best', 'leading', 'ahead', 'surges'
        }
        
        negative_words = {
            'bad', 'terrible', 'negative', 'fail', 'failure', 'lose', 'losing',
            'weak', 'decline', 'fall', 'crash', 'drop', 'loss', 'deficit',
            'concern', 'worry', 'pessimistic', 'fear', 'bearish', 'crisis',
            'down', 'low', 'decrease', 'worse', 'worst', 'behind', 'risk',
            'challenges', 'concerns'
        }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total = len(words)
        
        if total == 0:
            return {'score': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
        
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        
        # Sentiment score from -1 to 1
        score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        
        return {
            'score': score,
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': 1 - positive_ratio - negative_ratio
        }
    
    def extract_news_features(self, query: str, sources: List[str] = None) -> Dict:
        """Extract news features for ML model"""
        articles = self.search_news(query, sources, max_articles=20)
        
        features = {
            'article_count': len(articles),
            'avg_sentiment': 0,
            'sentiment_std': 0,
            'positive_ratio': 0,
            'negative_ratio': 0,
            'source_diversity': 0,
            'recency_score': 0
        }
        
        if not articles:
            return features
        
        # Analyze sentiments
        sentiments = []
        sources_set = set()
        
        for article in articles:
            sentiment = self.calculate_basic_sentiment(article['title'])
            sentiments.append(sentiment['score'])
            sources_set.add(article['source'])
        
        if sentiments:
            import statistics
            features['avg_sentiment'] = statistics.mean(sentiments)
            features['sentiment_std'] = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
            features['positive_ratio'] = sum(1 for s in sentiments if s > 0) / len(sentiments)
            features['negative_ratio'] = sum(1 for s in sentiments if s < 0) / len(sentiments)
        
        features['source_diversity'] = len(sources_set) / max(len(sources_set), 1)
        
        # Recency score
        recent_count = sum(1 for a in articles 
                          if any(term in a.get('published_date', '').lower() 
                                for term in ['hour', 'minute', 'today']))
        features['recency_score'] = recent_count / len(articles) if articles else 0
        
        return features


# For backward compatibility, create alias
NewsScraper = RobustNewsScraper


# Test
if __name__ == "__main__":
    print("="*60)
    print("ROBUST NEWS SCRAPER TEST")
    print("="*60)
    
    scraper = RobustNewsScraper(use_mock_fallback=True)
    
    test_queries = ["Trump", "Kawhi Leonard", "Bitcoin"]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        print("-"*60)
        
        articles = scraper.search_news(query, max_articles=5)
        print(f"Found {len(articles)} articles:")
        
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. [{article['source']}] {article['title']}")
            sentiment = scraper.calculate_basic_sentiment(article['title'])
            print(f"   Sentiment: {sentiment['score']:+.2f}")
        
        features = scraper.extract_news_features(query)
        print(f"\nFeatures: {features['article_count']} articles, "
              f"sentiment {features['avg_sentiment']:+.2f}, "
              f"{features['positive_ratio']:.0%} positive")
