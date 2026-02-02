"""
Example Usage Script
Demonstrates how to use all components of the Kalshi Market Prediction system
"""

import asyncio
import os
from datetime import datetime
import json

# Import our services
from kalshi_service import KalshiService
from news_scraper import NewsScraper
from gemini_analyzer import GeminiAnalyzer
from ml_classifier import MarketClassifier
from integration_service import MarketPredictionService


def example_1_basic_kalshi_usage():
    """Example 1: Basic Kalshi API interaction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Kalshi API Usage")
    print("="*60 + "\n")
    
    # Initialize service
    kalshi = KalshiService()
    
    # Get Trump markets
    print("Fetching Trump-related markets...")
    markets = kalshi.get_markets_by_category("Trump", limit=5)
    
    print(f"Found {len(markets)} markets:\n")
    for i, market in enumerate(markets, 1):
        print(f"{i}. {market['ticker']}")
        print(f"   Title: {market['title']}")
        print(f"   Price: ${market.get('last_price', 0)/100:.2f}")
        print()
    
    # Get detailed info for first market
    if markets:
        ticker = markets[0]['ticker']
        print(f"Getting details for {ticker}...")
        details = kalshi.get_market_details(ticker)
        
        if details:
            print(f"\nDetailed Market Info:")
            print(f"  Open Interest: {details.get('open_interest', 0):,}")
            print(f"  24h Volume: {details.get('volume_24h', 0):,}")
            print(f"  Status: {details.get('status')}")


def example_2_news_scraping():
    """Example 2: News scraping and sentiment analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: News Scraping & Sentiment")
    print("="*60 + "\n")
    
    scraper = NewsScraper()
    
    # Search for Trump news
    query = "Trump"
    print(f"Searching for '{query}' news...")
    articles = scraper.search_news(query, max_articles=10)
    
    print(f"\nFound {len(articles)} articles:\n")
    for i, article in enumerate(articles[:5], 1):
        print(f"{i}. [{article['source']}] {article['title'][:80]}...")
        
        # Calculate sentiment for this article
        sentiment = scraper.calculate_basic_sentiment(article['title'])
        print(f"   Sentiment: {sentiment['score']:.2f} "
              f"(Pos: {sentiment['positive']:.1%}, Neg: {sentiment['negative']:.1%})")
        print()
    
    # Get aggregated news features
    print("\nAggregated News Features:")
    features = scraper.extract_news_features(query)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


async def example_3_gemini_analysis():
    """Example 3: Advanced analysis with Gemini"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Gemini AI Analysis")
    print("="*60 + "\n")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Skipping this example.")
        print("   Get a key at: https://aistudio.google.com/app/apikey")
        return
    
    gemini = GeminiAnalyzer(api_key)
    scraper = NewsScraper()
    
    # Get some articles
    articles = scraper.search_news("Trump", max_articles=5)
    
    if not articles:
        print("No articles found. Skipping.")
        return
    
    # Analyze sentiment with Gemini
    print("Running Gemini sentiment analysis...")
    sentiment = gemini.analyze_sentiment(articles)
    
    print(f"\nGemini Sentiment Analysis:")
    print(f"  Overall: {sentiment['overall_sentiment']}")
    print(f"  Score: {sentiment['sentiment_score']:.2f}")
    print(f"  Confidence: {sentiment['confidence']:.1%}")
    print(f"  Market Impact: {sentiment.get('market_impact', 'N/A')}")
    print(f"\nKey Themes:")
    for theme in sentiment.get('key_themes', []):
        print(f"  - {theme}")
    print(f"\nSummary:")
    print(f"  {sentiment.get('summary', 'N/A')}")
    
    # Extract entities
    print("\n\nExtracting entities and events...")
    entities = gemini.extract_entities_and_events(articles)
    
    print(f"\nKey People:")
    for person in entities.get('key_people', [])[:5]:
        print(f"  - {person}")
    
    print(f"\nKey Events:")
    for event in entities.get('key_events', [])[:5]:
        print(f"  - {event}")


def example_4_ml_prediction():
    """Example 4: Machine Learning prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 4: ML Classification")
    print("="*60 + "\n")
    
    # Create sample data
    sample_market_features = {
        'current_price': 0.55,
        'volume_24h': 75000,
        'open_interest': 150000,
        'bid_ask_spread': 0.02,
        'days_to_expiration': 45,
        'price_change_7d': 0.12,  # 12% increase
        'volatility': 0.08,
        'volume_trend': 0.15  # 15% increase
    }
    
    sample_news_features = {
        'article_count': 18,
        'avg_sentiment': 0.35,  # Positive sentiment
        'sentiment_std': 0.15,
        'positive_ratio': 0.65,
        'negative_ratio': 0.20,
        'source_diversity': 0.75,
        'recency_score': 0.80
    }
    
    # Initialize and load pre-trained model
    classifier = MarketClassifier()
    
    try:
        classifier.load_model('trained_model.joblib')
        print("✓ Loaded pre-trained model\n")
    except:
        print("⚠️  No pre-trained model found. Training new model...\n")
        from ml_classifier import generate_synthetic_data
        
        # Train on synthetic data
        training_data, labels = generate_synthetic_data(200)
        metrics = classifier.train(training_data, labels)
        classifier.save_model('trained_model.joblib')
        print(f"✓ Model trained (Test Accuracy: {metrics['test_accuracy']:.3f})\n")
    
    # Make prediction
    print("Making prediction with sample data...")
    print("\nMarket Features:")
    for key, value in sample_market_features.items():
        print(f"  {key}: {value}")
    
    print("\nNews Features:")
    for key, value in sample_news_features.items():
        print(f"  {key}: {value}")
    
    prediction = classifier.predict(sample_market_features, sample_news_features)
    
    print(f"\n{'='*40}")
    print(f"PREDICTION: {prediction['prediction'].upper()}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"{'='*40}")
    
    print(f"\nClass Probabilities:")
    for cls, prob in sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls.ljust(15)}: {prob:.1%} {'█' * int(prob * 30)}")
    
    print(f"\nRecommendation:")
    print(f"  {prediction['recommendation']}")
    
    print(f"\nTop Contributing Features:")
    for feature in prediction.get('top_features', [])[:5]:
        print(f"  {feature['name']}: {feature['value']:.3f} (importance: {feature['importance']:.3f})")


async def example_5_full_integration():
    """Example 5: Full integration - end-to-end prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Full Integration Pipeline")
    print("="*60 + "\n")
    
    # Initialize service
    service = MarketPredictionService(
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Ensure model is trained
    if not service.classifier.is_trained:
        print("Training model...")
        service.train_model()
        print()
    
    # Get markets
    print("Fetching Trump markets...")
    markets = service.get_markets_by_category("Trump", limit=3)
    
    if not markets:
        print("No markets found. The Kalshi API might be rate-limiting.")
        return
    
    print(f"Found {len(markets)} markets\n")
    
    # Get full prediction for first market
    ticker = markets[0]['ticker']
    print(f"Getting full prediction for: {ticker}")
    print(f"Market: {markets[0]['title']}\n")
    
    print("Pipeline steps:")
    print("  1. Fetching market data...")
    print("  2. Scraping news articles...")
    print("  3. Running AI analysis...")
    print("  4. Generating prediction...\n")
    
    prediction = await service.get_full_prediction(ticker, category="Trump")
    
    if 'error' in prediction:
        print(f"Error: {prediction['error']}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nMarket: {prediction['market']['title']}")
    print(f"Current Price: ${prediction['market']['price']:.2f}")
    print(f"24h Volume: {prediction['market']['volume_24h']:,}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {prediction['prediction']['recommendation'].upper()}")
    print(f"CONFIDENCE: {prediction['prediction']['confidence']:.1%}")
    print(f"{'='*60}")
    
    print(f"\nProbabilities:")
    for cls, prob in prediction['prediction']['probabilities'].items():
        print(f"  {cls.ljust(15)}: {prob:.1%}")
    
    print(f"\nExplanation:")
    print(f"  {prediction['prediction']['explanation']}")
    
    print(f"\nNews Analysis:")
    print(f"  Articles analyzed: {prediction['news']['article_count']}")
    
    if prediction.get('news', {}).get('sentiment'):
        sentiment = prediction['news']['sentiment']
        print(f"  Overall sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
        print(f"  Sentiment score: {sentiment.get('sentiment_score', 0):.2f}")
    
    if prediction.get('context'):
        context = prediction['context']
        print(f"\nMarket Context:")
        print(f"  Event significance: {context.get('event_significance', 0)}/10")
        print(f"  Risk level: {context.get('risk_level', 'N/A')}")
    
    print(f"\nTop Features Influencing Prediction:")
    for i, feature in enumerate(prediction['analysis'].get('top_features', [])[:5], 1):
        print(f"  {i}. {feature['name']}: {feature['value']:.3f}")


async def example_6_batch_predictions():
    """Example 6: Batch predictions for multiple markets"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Predictions")
    print("="*60 + "\n")
    
    service = MarketPredictionService(
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Ensure model is trained
    if not service.classifier.is_trained:
        service.train_model()
    
    # Get multiple markets
    markets = service.get_markets_by_category("Trump", limit=3)
    
    if len(markets) < 2:
        print("Not enough markets for batch prediction.")
        return
    
    tickers = [m['ticker'] for m in markets[:2]]
    
    print(f"Getting predictions for {len(tickers)} markets...")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    # Batch predict
    predictions = await service.batch_predict(tickers)
    
    # Display results
    for pred in predictions:
        if 'error' in pred:
            print(f"\n{pred['ticker']}: Error - {pred['error']}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Market: {pred['market']['title']}")
        print(f"Ticker: {pred['ticker']}")
        print(f"Recommendation: {pred['prediction']['recommendation'].upper()}")
        print(f"Confidence: {pred['prediction']['confidence']:.1%}")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("KALSHI MARKET PREDICTION - EXAMPLES")
    print("="*60)
    
    # Run examples
    try:
        example_1_basic_kalshi_usage()
        input("\nPress Enter to continue to next example...")
        
        example_2_news_scraping()
        input("\nPress Enter to continue to next example...")
        
        await example_3_gemini_analysis()
        input("\nPress Enter to continue to next example...")
        
        example_4_ml_prediction()
        input("\nPress Enter to continue to next example...")
        
        await example_5_full_integration()
        input("\nPress Enter to continue to next example...")
        
        await example_6_batch_predictions()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
