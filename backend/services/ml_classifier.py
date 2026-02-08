"""
FINAL Training Pipeline - Complete & Production Ready
Combines market features + enhanced news features
"""

from kalshi_service import KalshiService
from news_scraper import RobustNewsScraper
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def extract_multiple_queries_from_market(market):
    """Extract multiple search queries from market for better news coverage"""
    
    title = market.get('title', '').lower()
    ticker = market.get('ticker', '')
    original_title = market.get('title', '')
    
    queries = []
    
    # Strategy 1: Direct keyword matching
    if 'trump' in title:
        queries.extend(['Trump', 'Donald Trump', 'Trump news'])
    
    if 'bitcoin' in title or 'btc' in title or 'crypto' in title:
        queries.extend(['Bitcoin', 'BTC', 'cryptocurrency'])
    
    if 'election' in title or 'senate' in title or 'congress' in title:
        queries.extend(['election 2026', 'senate race', 'midterm elections'])
    
    if 'federal reserve' in title or 'fed' in title or 'interest rate' in title:
        queries.extend(['Federal Reserve', 'Fed interest rates', 'Jerome Powell'])
    
    # Strategy 2: Sports-specific queries
    sports_map = {
        'nba': ['NBA', 'basketball'],
        'nfl': ['NFL', 'football'],
        'mlb': ['MLB', 'baseball'],
        'nhl': ['NHL', 'hockey'],
        'fifa': ['FIFA', 'World Cup'],
        'uefa': ['UEFA', 'Champions League'],
    }
    
    for sport, sport_queries in sports_map.items():
        if sport in title:
            queries.extend(sport_queries)
            
            # Try to extract team names (look for vs, versus, against)
            for separator in ['vs', 'versus', 'against', 'v.']:
                if separator in title:
                    parts = title.split(separator)
                    if len(parts) >= 2:
                        team1 = parts[0].strip().split()[-1]  # Last word before separator
                        team2 = parts[1].strip().split()[0]   # First word after separator
                        if len(team1) > 2:
                            queries.append(team1)
                        if len(team2) > 2:
                            queries.append(team2)
    
    # Strategy 3: Extract proper nouns (capitalized words)
    words = original_title.split()
    
    for word in words:
        if word and len(word) > 2 and word[0].isupper():
            clean = word.strip(',:;.()[]')
            # Skip common words
            if clean.lower() not in ['yes', 'no', 'and', 'or', 'the', 'will', 'over', 'under']:
                queries.append(clean)
    
    # Strategy 4: Company/stock mentions
    if any(term in title for term in ['stock', 'share', 'nasdaq', 'nyse', 'sp500', 's&p']):
        # Extract company names (usually capitalized sequences)
        words_list = original_title.split()
        for i, word in enumerate(words_list):
            if word and word[0].isupper() and len(word) > 3:
                queries.append(word.strip(',:;.'))
    
    # Strategy 5: Fallback - use first meaningful phrase
    if not queries:
        # Take first 3-4 non-common words
        common_words = {'yes', 'no', 'will', 'the', 'and', 'or', 'over', 'under', 'more', 'less', 'than'}
        meaningful_words = [w for w in words[:10] if w.lower() not in common_words and len(w) > 2]
        if meaningful_words:
            queries.append(' '.join(meaningful_words[:3]))
        else:
            # Absolute fallback
            queries.append(' '.join(words[:3]))
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen and len(q_lower) > 1:
            seen.add(q_lower)
            unique.append(q.strip())
    
    return unique[:5]  # Return max 5 queries


def extract_all_features(kalshi, scraper, market):
    """Extract comprehensive market + news features"""
    
    ticker = market.get('ticker', '')
    title = market.get('title', '')
    
    print(f"    {ticker[:60]}")
    
    features = {
        'ticker': ticker,
        'title': title,
        'result': market.get('result', ''),
    }
    
    # ==================== MARKET FEATURES ====================
    
    # Price features
    last_price = market.get('last_price', 0)
    features['final_price'] = last_price / 100 if last_price else None
    
    previous_price = market.get('previous_price', 0)
    features['previous_price'] = previous_price / 100 if previous_price else None
    
    if features['final_price'] is not None and features['previous_price'] is not None:
        features['price_change'] = features['final_price'] - features['previous_price']
    else:
        features['price_change'] = None
    
    # Bid/Ask spread
    yes_bid = market.get('yes_bid', 0) or market.get('previous_yes_bid', 0)
    yes_ask = market.get('yes_ask', 0) or market.get('previous_yes_ask', 0)
    
    if yes_bid and yes_ask:
        features['final_spread'] = (yes_ask - yes_bid) / 100
        features['spread_pct'] = ((yes_ask - yes_bid) / yes_ask) if yes_ask > 0 else None
    else:
        features['final_spread'] = None
        features['spread_pct'] = None
    
    # Volume features
    features['total_volume'] = market.get('volume', 0)
    features['volume_24h'] = market.get('volume_24h', 0)
    features['open_interest'] = market.get('open_interest', 0)
    features['liquidity'] = market.get('liquidity', 0)
    
    # Liquidity metrics
    if features['total_volume'] and features['open_interest']:
        features['volume_to_oi_ratio'] = features['total_volume'] / max(features['open_interest'], 1)
    else:
        features['volume_to_oi_ratio'] = None
    
    # Time features
    created_time = market.get('created_time')
    close_time = market.get('close_time')
    
    if created_time and close_time:
        try:
            created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            close_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            
            lifespan = (close_dt - created_dt).total_seconds() / 86400
            features['market_lifespan_days'] = lifespan
            
            # How recently did it close?
            now = datetime.now(timezone.utc)
            days_since = (now - close_dt).total_seconds() / 86400
            features['days_since_close'] = days_since
            
        except:
            features['market_lifespan_days'] = None
            features['days_since_close'] = None
    else:
        features['market_lifespan_days'] = None
        features['days_since_close'] = None
    
    # Market type
    features['is_multivariate'] = 1 if 'KXMV' in ticker else 0
    
    # Historical price data (if available)
    try:
        history = kalshi.get_market_history(ticker, days=365)
        
        if not history.empty and 'price' in history.columns and len(history) > 1:
            prices = history['price'] / 100
            returns = prices.pct_change().dropna()
            
            features['price_volatility'] = returns.std() if len(returns) > 0 else 0
            features['price_range'] = prices.max() - prices.min()
            features['avg_price'] = prices.mean()
            features['num_price_updates'] = len(history)
            
            # Price momentum
            if len(prices) >= 7:
                features['price_momentum_7d'] = prices.iloc[-1] - prices.iloc[-7]
            else:
                features['price_momentum_7d'] = None
        else:
            features['price_volatility'] = None
            features['price_range'] = None
            features['avg_price'] = None
            features['num_price_updates'] = 0
            features['price_momentum_7d'] = None
            
    except Exception as e:
        features['price_volatility'] = None
        features['price_range'] = None
        features['avg_price'] = None
        features['num_price_updates'] = 0
        features['price_momentum_7d'] = None
    
    # ==================== NEWS FEATURES ====================
    
    # Extract multiple queries for better coverage
    queries = extract_multiple_queries_from_market(market)
    
    print(f"      Queries: {queries[:3]}")
    
    try:
        news_features = scraper.extract_enhanced_news_features(queries)
        features.update(news_features)
        
        print(f"      → {news_features['news_article_count']} articles, "
              f"sentiment: {news_features['news_avg_sentiment']:+.2f}, "
              f"sources: {news_features['news_source_diversity']}")
        
    except Exception as e:
        print(f"      ✗ News error: {e}")
        # Default news features
        features.update({
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
        })
    
    return features


def fetch_settled_markets(kalshi):
    """Fetch settled markets from Kalshi API"""
    
    print("Fetching settled markets from Kalshi...")
    
    settled = []
    
    for status in ['settled', 'closed', 'finalized']:
        try:
            response = kalshi.session.get(
                f"{kalshi.BASE_URL}/markets",
                params={'limit': 200, 'status': status},
                timeout=10
            )
            
            if response.ok:
                markets = response.json().get('markets', [])
                print(f"  ✓ {len(markets):3d} markets with status '{status}'")
                settled.extend(markets)
            else:
                print(f"  ✗ Failed to fetch '{status}' markets: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Error fetching '{status}': {e}")
    
    # Remove duplicates
    seen = set()
    unique_settled = []
    
    for m in settled:
        ticker = m.get('ticker')
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique_settled.append(m)
    
    print(f"\n  → Total: {len(unique_settled)} unique settled markets\n")
    
    return unique_settled


def analyze_feature_quality(df):
    """Analyze and select high-quality features"""
    
    print("\n" + "="*70)
    print("FEATURE QUALITY ANALYSIS")
    print("="*70 + "\n")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    good_features = []
    
    print(f"{'Feature':<35} {'Non-null':>10} {'Unique':>8} {'Std':>10} {'Status':>8}")
    print("-" * 70)
    
    for col in numeric_cols:
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df)) * 100
        unique_count = df[col].nunique()
        unique_pct = (unique_count / non_null) * 100 if non_null > 0 else 0
        std = df[col].std()
        
        # Feature selection criteria
        is_good = (
            (unique_pct >= 20 or std > 0.1) and  # Has variance
            non_null_pct >= 50                     # Mostly non-null
        )
        
        if is_good:
            good_features.append(col)
            status = "✓ GOOD"
        else:
            status = "✗ Skip"
        
        print(f"{col:<35} {non_null_pct:>9.1f}% {unique_pct:>7.1f}% {std:>10.4f} {status:>8}")
    
    print("\n" + "="*70)
    print(f"SELECTED: {len(good_features)} features")
    print("="*70 + "\n")
    
    # Categorize features
    market_features = [f for f in good_features if not f.startswith('news_')]
    news_features = [f for f in good_features if f.startswith('news_')]
    
    print(f"Market features ({len(market_features)}):")
    for feat in market_features:
        print(f"  • {feat}")
    
    print(f"\nNews features ({len(news_features)}):")
    for feat in news_features:
        print(f"  • {feat}")
    
    return good_features


def train_model(df, features):
    """Train and evaluate classification model"""
    
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70 + "\n")
    
    # Prepare data
    X = df[features].copy()
    y = df['result'].copy()
    
    # Remove rows with missing data
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"Dataset size: {len(X)} samples × {len(features)} features")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print()
    
    if len(y.unique()) < 2:
        print("❌ Error: Need both 'yes' and 'no' outcomes!")
        return None
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples\n")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=min(5, len(X_train) // 2),
        n_jobs=-1
    )
    
    print(f"  CV accuracy:    {cv_scores.mean():.1%} ± {cv_scores.std():.1%}\n")
    
    # Feature importance
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("="*70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*70 + "\n")
    
    for i, (feat, imp) in enumerate(feature_importance[:15], 1):
        feat_type = "NEWS  " if feat.startswith('news_') else "MARKET"
        bar = "█" * int(imp * 50)
        print(f"{i:2d}. [{feat_type}] {feat:<35} {imp:.4f} {bar}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70 + "\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y.unique())
    
    print("Confusion Matrix:\n")
    print("Predicted →     ", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print("\nActual ↓")
    
    for i, label in enumerate(labels):
        print(f"{label:>16}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def main():
    print("="*70)
    print("KALSHI MARKET PREDICTOR - FINAL TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Initialize services
    kalshi = KalshiService()
    scraper = RobustNewsScraper()
    
    # Fetch settled markets
    settled_markets = fetch_settled_markets(kalshi)
    
    if not settled_markets:
        print("❌ No settled markets found! Cannot train model.")
        return
    
    # Filter for volume (optional)
    active_markets = [m for m in settled_markets if m.get('volume', 0) >= 10]
    
    if not active_markets:
        print("⚠️  No markets with volume >= 10, using all settled markets\n")
        active_markets = settled_markets
    else:
        print(f"✓ Filtered to {len(active_markets)} markets with volume >= 10\n")
    
    # Limit to 50 markets to avoid excessive API calls
    markets_to_process = active_markets[:50]
    
    print(f"Processing {len(markets_to_process)} markets...\n")
    print("="*70 + "\n")
    
    # Extract features
    features_list = []
    
    for i, market in enumerate(markets_to_process, 1):
        print(f"[{i}/{len(markets_to_process)}]")
        
        try:
            features = extract_all_features(kalshi, scraper, market)
            features_list.append(features)
        except Exception as e:
            print(f"      ✗ Error: {e}")
        
        time.sleep(0.5)  # Rate limiting
        print()
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Filter for valid results
    df = df[df['result'].notna() & (df['result'] != '')]
    
    print("="*70)
    print(f"DATASET CREATED: {len(df)} markets with valid results")
    print("="*70)
    
    if len(df) < 20:
        print(f"\n❌ Error: Only {len(df)} markets with results")
        print("   Need at least 20 for reliable training.")
        return
    
    # Analyze and select features
    good_features = analyze_feature_quality(df)
    
    if len(good_features) == 0:
        print("\n❌ Error: No usable features found!")
        return
    
    # Train model
    result = train_model(df, good_features)
    
    if not result:
        print("\n❌ Training failed!")
        return
    
    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70 + "\n")
    
    model_data = {
        'model': result['model'],
        'scaler': result['scaler'],
        'features': result['features'],
        'metadata': {
            'train_accuracy': result['train_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'num_features': len(result['features']),
            'num_samples': len(df),
            'trained_at': datetime.now().isoformat()
        }
    }
    
    with open('kalshi_final_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✓ Model saved to 'kalshi_final_model.pkl'")
    
    # Save dataset
    df.to_csv('training_data_final.csv', index=False)
    print("✓ Training data saved to 'training_data_final.csv'")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {result['test_accuracy']:.1%}")
    print(f"Features used: {len(result['features'])}")
    print(f"  - Market features: {len([f for f in result['features'] if not f.startswith('news_')])}")
    print(f"  - News features: {len([f for f in result['features'] if f.startswith('news_')])}")


if __name__ == "__main__":
    main()