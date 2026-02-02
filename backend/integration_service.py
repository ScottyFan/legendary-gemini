"""
Main Integration Service
Orchestrates all components: Kalshi API, News Scraper, Gemini, and ML Classifier
"""

from kalshi_service import KalshiService
from news_scraper import NewsScraper
# from gemini_analyzer import GeminiAnalyzer
from ml_classifier import MarketClassifier
from typing import Dict, List, Optional
import asyncio


class MarketPredictionService:
    """Main service that integrates all components"""
    
    def __init__(self, 
                 kalshi_email: str = None,
                 kalshi_password: str = None,
                 gemini_api_key: str = None):
        """
        Initialize the prediction service
        
        Args:
            kalshi_email: Kalshi account email (optional)
            kalshi_password: Kalshi account password (optional)
            gemini_api_key: Google Gemini API key (optional)
        """
        self.kalshi = KalshiService(kalshi_email, kalshi_password)
        self.scraper = NewsScraper()
        self.gemini = None
        self.classifier = MarketClassifier(model_type='random_forest')
        
        # Load pre-trained model if available
        try:
            self.classifier.load_model('trained_model.joblib')
            print("Loaded pre-trained model")
        except:
            print("No pre-trained model found. Train a new model before making predictions.")
    
    def get_categories(self) -> List[str]:
        """Get available market categories"""
        return ['Politics', 'Sports', 'Culture', 'Crypto', 'Trump']
    
    def get_markets_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """
        Get markets for a specific category
        
        Args:
            category: Category name
            limit: Maximum number of markets
            
        Returns:
            List of market summaries
        """
        markets = self.kalshi.get_markets_by_category(category, limit)
        
        # Format for frontend
        return [
            {
                'ticker': m['ticker'],
                'title': m['title'],
                'price': m.get('last_price', 0) / 100,  # Convert cents to dollars
                'volume': m.get('volume_24h', 0),
                'close_time': m.get('close_time')
            }
            for m in markets
        ]
    
    async def get_full_prediction(self, 
                                  market_ticker: str,
                                  category: str = None) -> Dict:
        """
        Get comprehensive prediction for a market
        
        Args:
            market_ticker: Kalshi market ticker
            category: Optional category hint for better news search
            
        Returns:
            Complete prediction analysis
        """
        # Step 1: Get market details and features
        print(f"Fetching market data for {market_ticker}...")
        market_details = self.kalshi.get_market_details(market_ticker)
        
        if not market_details:
            return {
                'error': 'Market not found',
                'ticker': market_ticker
            }
        
        market_features = self.kalshi.extract_market_features(market_ticker)
        
        # Step 2: Determine search query from market title
        market_title = market_details.get('title', '')
        search_query = self._extract_search_query(market_title, category)
        
        print(f"Searching news for: {search_query}")
        
        # Step 3: Scrape news
        articles = self.scraper.search_news(search_query, max_articles=20)
        news_features = self.scraper.extract_news_features(search_query)
        
        # Step 4: Use Gemini for enhanced analysis (if available)
        gemini_features = {}
        gemini_sentiment = None
        gemini_context = None
        
        if self.gemini and articles:
            print("Running Gemini analysis...")
            gemini_sentiment = self.gemini.analyze_sentiment(articles)
            
            gemini_context = self.gemini.analyze_market_context(
                market_title,
                market_details.get('description', ''),
                gemini_sentiment.get('summary', '')
            )
            
            # Extract numerical features for ML
            gemini_features = {
                'sentiment_score': gemini_sentiment.get('sentiment_score', 0),
                'confidence': gemini_sentiment.get('confidence', 0),
                'event_significance': gemini_context.get('event_significance', 5) / 10
            }
        
        # Step 5: Make ML prediction
        print("Generating prediction...")
        prediction = self.classifier.predict(
            market_features,
            news_features,
            gemini_features if gemini_features else None
        )
        
        # Step 6: Generate explanation using Gemini (if available)
        explanation = prediction.get('recommendation', '')
        
        if self.gemini:
            explanation = self.gemini.generate_prediction_explanation(
                market_title,
                prediction['prediction'],
                prediction['confidence'],
                market_features,
                news_features
            )
        
        # Step 7: Compile full response
        return {
            'ticker': market_ticker,
            'market': {
                'title': market_title,
                'price': market_details.get('last_price', 0) / 100,
                'volume_24h': market_details.get('volume_24h', 0),
                'close_time': market_details.get('close_time')
            },
            'prediction': {
                'recommendation': prediction['prediction'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'explanation': explanation
            },
            'analysis': {
                'market_features': market_features,
                'news_features': news_features,
                'top_features': prediction.get('top_features', [])
            },
            'news': {
                'article_count': len(articles),
                'articles': articles[:5],  # Return top 5 articles
                'sentiment': gemini_sentiment if gemini_sentiment else None
            },
            'context': gemini_context if gemini_context else None
        }
    
    def _extract_search_query(self, market_title: str, category: str = None) -> str:
        """
        Extract relevant search query from market title
        
        Args:
            market_title: Market title
            category: Optional category hint
            
        Returns:
            Search query string
        """
        # Remove common market-specific words
        stop_words = [
            'will', 'be', 'by', 'in', 'on', 'at', 'to', 'for',
            'market', 'prediction', 'outcome', 'result'
        ]
        
        words = market_title.lower().split()
        
        # Keep important words
        query_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # If we have a category hint, include it
        if category and category.lower() not in ' '.join(query_words).lower():
            query_words.insert(0, category)
        
        # Take first 3-4 most relevant words
        return ' '.join(query_words[:4])
    
    def train_model(self, training_data_path: str = None):
        """
        Train the ML classifier
        
        Args:
            training_data_path: Path to training data CSV
        """
        # TODO: Implement training data loading and model training
        # For now, generate synthetic data
        print("Training model on synthetic data...")
        
        from ml_classifier import generate_synthetic_data
        training_data, labels = generate_synthetic_data(500)
        
        metrics = self.classifier.train(training_data, labels)
        
        print(f"\nTraining Complete!")
        print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        # Save trained model
        self.classifier.save_model('trained_model.joblib')
        print("Model saved to trained_model.joblib")
        
        return metrics
    
    async def batch_predict(self, market_tickers: List[str]) -> List[Dict]:
        """
        Get predictions for multiple markets
        
        Args:
            market_tickers: List of market tickers
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for ticker in market_tickers:
            try:
                pred = await self.get_full_prediction(ticker)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting {ticker}: {e}")
                predictions.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return predictions


# Example usage
if __name__ == "__main__":
    import os
    import json
    
    # Initialize service
    service = MarketPredictionService(
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Train model (first time setup)
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    metrics = service.train_model()
    
    # Get categories
    print("\n" + "=" * 60)
    print("AVAILABLE CATEGORIES")
    print("=" * 60)
    categories = service.get_categories()
    print(categories)
    
    # Get Trump markets
    print("\n" + "=" * 60)
    print("TRUMP MARKETS")
    print("=" * 60)
    trump_markets = service.get_markets_by_category("Trump", limit=5)
    print(f"Found {len(trump_markets)} Trump markets")
    
    for market in trump_markets:
        print(f"  - {market['ticker']}: {market['title']}")
    
    # Get prediction for first market
    if trump_markets:
        print("\n" + "=" * 60)
        print("GETTING PREDICTION")
        print("=" * 60)
        
        ticker = trump_markets[0]['ticker']
        
        # Use asyncio to run async function
        async def main():
            prediction = await service.get_full_prediction(ticker, category="Trump")
            print(json.dumps(prediction, indent=2, default=str))
        
        asyncio.run(main())
