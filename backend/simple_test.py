"""
Simple Test Script - Works immediately without any API keys!
Just run: python simple_test.py
"""

import asyncio
import sys

# Add current directory to path
sys.path.insert(0, '.')

from integration_service import MarketPredictionService


async def main():
    print("="*60)
    print("KALSHI MARKET PREDICTOR - SIMPLE TEST")
    print("="*60)
    print("\nNo API keys needed! Using public Kalshi data.\n")
    
    # Initialize service (no credentials needed!)
    print("Initializing service...")
    service = MarketPredictionService()
    
    # Train model if needed
    if not service.classifier.is_trained:
        print("\nTraining ML model on synthetic data...")
        metrics = service.train_model()
        print(f"✓ Model trained! Test accuracy: {metrics['test_accuracy']:.1%}\n")
    
    # Get Trump markets
    print("Fetching Trump-related markets from Kalshi...")
    try:
        markets = service.get_markets_by_category("Trump", limit=3)
        
        if not markets:
            print("⚠️  No markets found. Kalshi API might be temporarily unavailable.")
            print("   This is normal - just try again in a few seconds.")
            return
        
        print(f"✓ Found {len(markets)} markets:\n")
        for i, market in enumerate(markets, 1):
            print(f"{i}. {market['ticker']}")
            print(f"   {market['title']}")
            print(f"   Current price: ${market['price']:.2f}")
            print(f"   24h volume: {market['volume']:,}")
            print()
        
        # Get prediction for first market
        selected = markets[0]
        print(f"Getting AI prediction for: {selected['ticker']}")
        print(f"Market: {selected['title']}\n")
        
        prediction = await service.get_full_prediction(
            selected['ticker'], 
            category="Trump"
        )
        
        if 'error' in prediction:
            print(f"Error: {prediction['error']}")
            return
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60 + "\n")
        
        rec = prediction['prediction']['recommendation']
        conf = prediction['prediction']['confidence']
        
        # Color coding for terminal
        if 'buy' in rec:
            symbol = "📈 BUY"
        elif 'sell' in rec:
            symbol = "📉 SELL"
        else:
            symbol = "➖ HOLD"
        
        print(f"{symbol} {rec.upper().replace('_', ' ')}")
        print(f"Confidence: {conf:.1%}")
        print()
        
        print("Class Probabilities:")
        for cls, prob in sorted(
            prediction['prediction']['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            bar = "█" * int(prob * 30)
            print(f"  {cls.replace('_', ' ').ljust(12)}: {prob:6.1%} {bar}")
        
        print(f"\nExplanation:")
        print(f"  {prediction['prediction']['explanation']}")
        
        print(f"\nMarket Analysis:")
        market_f = prediction['analysis']['market_features']
        print(f"  Price change (7d): {market_f.get('price_change_7d', 0):+.1%}")
        print(f"  Volatility: {market_f.get('volatility', 0):.3f}")
        print(f"  Volume trend: {market_f.get('volume_trend', 0):+.1%}")
        
        print(f"\nNews Analysis:")
        news_f = prediction['analysis']['news_features']
        print(f"  Articles analyzed: {news_f.get('article_count', 0)}")
        print(f"  Average sentiment: {news_f.get('avg_sentiment', 0):+.2f}")
        print(f"  Positive ratio: {news_f.get('positive_ratio', 0):.1%}")
        
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be due to:")
        print("  1. Kalshi API rate limiting (wait a few seconds)")
        print("  2. Network connectivity issues")
        print("  3. Missing Python packages (run: pip install -r requirements.txt)")
        import traceback
        print("\nFull error:")
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting Kalshi Market Predictor...\n")
    asyncio.run(main())
    print("\n✨ Thanks for testing!\n")
