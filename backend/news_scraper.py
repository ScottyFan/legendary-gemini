"""
News Scraper Service
Web scraping and sentiment analysis for market-related news
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, quote_plus
import time


class NewsScraper:
    """Scrape news articles and extract sentiment features"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        })
        
        # News sources configuration
        self.sources = {
            'cnn': {
                'search_url': 'https://www.cnn.com/search?q={query}&from=0&size=10&page=1',
                'base_url': 'https://www.cnn.com'
            },
            'reuters': {
                # Updated Reuters search URL and handling
                'search_url': 'https://www.reuters.com/site-search/?query={query}',
                'base_url': 'https://www.reuters.com'
            },
            'bbc': {
                'search_url': 'https://www.bbc.com/search?q={query}',
                'base_url': 'https://www.bbc.com'
            }
        }
    
    def search_news(self, query: str, sources: List[str] = None, max_articles: int = 20) -> List[Dict]:
        """
        Search for news articles across multiple sources
        
        Args:
            query: Search query (e.g., "Trump", "Bitcoin", etc.)
            sources: List of source names to search (default: all)
            max_articles: Maximum number of articles to return
            
        Returns:
            List of article dictionaries
        """
        if sources is None:
            sources = list(self.sources.keys())
            
        # Clean query: remove special chars that might break URLs
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        # Collapse multiple spaces
        clean_query = re.sub(r'\s+', ' ', clean_query)
        
        all_articles = []
        
        for source in sources:
            if source not in self.sources:
                continue
            
            try:
                articles = self._search_source(source, clean_query)
                all_articles.extend(articles)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                # print(f"Error scraping {source}: {e}") # Reduce noise
                pass
        
        # Sort by date (most recent first) and limit
        all_articles.sort(key=lambda x: x.get('published_date', ''), reverse=True)
        return all_articles[:max_articles]
    
    def _search_source(self, source: str, query: str) -> List[Dict]:
        """
        Search a specific news source
        
        Args:
            source: Source name
            query: Search query
            
        Returns:
            List of articles from that source
        """
        articles = []
        
        try:
            search_url = self.sources[source]['search_url'].format(query=quote_plus(query))
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Source-specific parsing
            if source == 'cnn':
                articles = self._parse_cnn(soup, query)
            elif source == 'reuters':
                articles = self._parse_reuters(soup, query)
            elif source == 'bbc':
                articles = self._parse_bbc(soup, query)
            
        except Exception as e:
            print(f"Error parsing {source}: {e}")
        
        return articles
    
    def _parse_cnn(self, soup: BeautifulSoup, query: str) -> List[Dict]:
        """Parse CNN search results"""
        articles = []
        
        # CNN uses specific article containers
        article_containers = soup.find_all('div', class_='cnn-search__result')
        
        for container in article_containers[:10]:
            try:
                title_elem = container.find('h3', class_='cnn-search__result-headline')
                link_elem = container.find('a', class_='cnn-search__result-link')
                date_elem = container.find('div', class_='cnn-search__result-publish-date')
                
                if title_elem and link_elem:
                    article = {
                        'source': 'CNN',
                        'title': title_elem.get_text(strip=True),
                        'url': urljoin(self.sources['cnn']['base_url'], link_elem['href']),
                        'published_date': date_elem.get_text(strip=True) if date_elem else '',
                        'query': query
                    }
                    articles.append(article)
            except Exception as e:
                continue
        
        return articles
    
    def _parse_reuters(self, soup: BeautifulSoup, query: str) -> List[Dict]:
        """Parse Reuters search results"""
        articles = []
        
        # Reuters article structure
        article_containers = soup.find_all('div', class_='search-result-indiv')
        
        for container in article_containers[:10]:
            try:
                link_elem = container.find('a')
                title_elem = container.find('h3')
                date_elem = container.find('time')
                
                if title_elem and link_elem:
                    article = {
                        'source': 'Reuters',
                        'title': title_elem.get_text(strip=True),
                        'url': urljoin(self.sources['reuters']['base_url'], link_elem['href']),
                        'published_date': date_elem['datetime'] if date_elem and date_elem.get('datetime') else '',
                        'query': query
                    }
                    articles.append(article)
            except Exception as e:
                continue
        
        return articles
    
    def _parse_bbc(self, soup: BeautifulSoup, query: str) -> List[Dict]:
        """Parse BBC search results"""
        articles = []
        
        # BBC search results
        article_containers = soup.find_all('article')
        
        for container in article_containers[:10]:
            try:
                link_elem = container.find('a')
                title_elem = container.find('h1') or container.find('h2')
                
                if title_elem and link_elem:
                    article = {
                        'source': 'BBC',
                        'title': title_elem.get_text(strip=True),
                        'url': urljoin(self.sources['bbc']['base_url'], link_elem['href']),
                        'published_date': '',
                        'query': query
                    }
                    articles.append(article)
            except Exception as e:
                continue
        
        return articles
    
    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch full article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Article text content or None
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main article content
            content = None
            
            # Common article content selectors
            selectors = [
                'article',
                '[class*="article-body"]',
                '[class*="story-body"]',
                '[class*="content-body"]',
                'main'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Fallback to all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            return content
        except Exception as e:
            print(f"Error fetching article content: {e}")
            return None
    
    def calculate_basic_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate basic sentiment using keyword matching
        (Simple implementation - use Gemini for better results)
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores dictionary
        """
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'success', 'win', 'winning',
            'strong', 'growth', 'rise', 'surge', 'rally', 'gain', 'profit',
            'boost', 'improve', 'optimistic', 'confidence', 'bullish'
        }
        
        negative_words = {
            'bad', 'terrible', 'negative', 'fail', 'failure', 'lose', 'losing',
            'weak', 'decline', 'fall', 'crash', 'drop', 'loss', 'deficit',
            'concern', 'worry', 'pessimistic', 'fear', 'bearish', 'crisis'
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
        """
        Extract comprehensive news features for ML prediction
        
        Args:
            query: Topic to search for
            sources: News sources to use
            
        Returns:
            Dictionary of news-based features
        """
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
        
        # Analyze headlines
        sentiments = []
        sources_set = set()
        
        for article in articles:
            # Basic sentiment on title
            title_sentiment = self.calculate_basic_sentiment(article['title'])
            sentiments.append(title_sentiment['score'])
            sources_set.add(article['source'])
        
        if sentiments:
            import statistics
            features['avg_sentiment'] = statistics.mean(sentiments)
            features['sentiment_std'] = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
            features['positive_ratio'] = sum(1 for s in sentiments if s > 0) / len(sentiments)
            features['negative_ratio'] = sum(1 for s in sentiments if s < 0) / len(sentiments)
        
        features['source_diversity'] = len(sources_set) / len(self.sources)
        
        # Calculate recency (articles from last 24h get higher score)
        recent_count = 0
        for article in articles:
            # Simple check if date string contains "hour" or "minute"
            date_str = article.get('published_date', '').lower()
            if 'hour' in date_str or 'minute' in date_str or 'today' in date_str:
                recent_count += 1
        
        features['recency_score'] = recent_count / len(articles) if articles else 0
        
        return features


# Example usage
if __name__ == "__main__":
    scraper = NewsScraper()
    
    # Search for Trump-related news
    print("Searching for Trump news...")
    articles = scraper.search_news("Trump", max_articles=5)
    
    print(f"\nFound {len(articles)} articles:")
    for article in articles:
        print(f"\n{article['source']}: {article['title'][:100]}")
    
    # Extract features
    print("\n\nExtracting news features...")
    features = scraper.extract_news_features("Trump")
    print("\nNews Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
