"""
Kalshi API Service - Ultra-Simple Version
Minimal parameters to avoid 400 errors
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd


class KalshiService:
    """Kalshi service"""
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, email: str = None, password: str = None):
        """Initialize Kalshi service"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        })
        self.token = None
        
        if email and password:
            self.login(email, password)
    
    def login(self, email: str, password: str) -> bool:
        """Authenticate with Kalshi API"""
        try:
            response = self.session.post(
                f"{self.BASE_URL}/login",
                json={"email": email, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data.get("token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    
    def _fetch_markets_simple(self, limit: int = 100) -> List[Dict]:
        """Fetch markets with minimal parameters"""
        
        # Try different API calls to find what works
        attempts = [
            # Attempt 1: Just limit
            {"limit": limit},
            # Attempt 2: No parameters at all
            {},
            # Attempt 3: Small limit
            {"limit": 10},
            # Attempt 4: Cursor-based (if they use pagination)
            {"limit": 20, "cursor": ""},
        ]
        
        for i, params in enumerate(attempts):
            try:
                print(f"  Attempt {i+1}: params={params}")
                
                response = self.session.get(
                    f"{self.BASE_URL}/markets",
                    params=params,
                    timeout=10
                )
                
                if response.ok:
                    data = response.json()
                    markets = data.get("markets", [])
                    if markets:
                        print(f"  ✓ Success with params={params}")
                        return markets
                else:
                    print(f"  ✗ HTTP {response.status_code}")
                    if response.status_code == 400:
                        print(f"    Error: {response.text[:200]}")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        return []
    
    def get_markets_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """Fetch markets and filter by category"""
        
        print(f"Fetching markets from Kalshi API...")
        
        try:
            # Get all available markets
            all_markets = self._fetch_markets_simple(limit=100)
            
            if not all_markets:
                print("  ⚠ Could not fetch any markets")
                return []
            
            print(f"  → Got {len(all_markets)} total markets")
            
            # Filter by status (accept 'active' or 'open')
            valid_markets = [
                m for m in all_markets 
                if m.get('status') in ['active', 'open', 'closed', None]  # Accept most
            ]
            
            print(f"  → {len(valid_markets)} markets after status filter")
            
            # Category filtering with broad keywords
            category_lower = category.lower()
            filtered_markets = []
            
            for market in valid_markets:
                title = market.get('title', '').lower()
                ticker = market.get('ticker', '').lower()
                
                matched = False
                
                if category_lower == 'trump':
                    if 'trump' in title or 'trump' in ticker:
                        matched = True
                
                elif category_lower == 'politics':
                    kw = ['election', 'president', 'senate', 'congress', 'vote', 
                          'poll', 'democrat', 'republican', 'biden', 'trump']
                    if any(k in title or k in ticker for k in kw):
                        matched = True
                
                elif category_lower == 'sports':
                    kw = ['nfl', 'nba', 'mlb', 'nhl', 'super', 'bowl', 'playoff',
                          'game', 'player', 'team', 'arsenal', 'manchester', 'barcelona',
                          'touchdown', 'assist', 'rebound', 'goal', 'score']
                    if any(k in title or k in ticker for k in kw):
                        matched = True
                
                elif category_lower in ['crypto', 'bitcoin']:
                    kw = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto']
                    if any(k in title or k in ticker for k in kw):
                        matched = True
                
                elif category_lower == 'culture':
                    kw = ['movie', 'film', 'oscar', 'grammy', 'music', 'award', 'album']
                    if any(k in title or k in ticker for k in kw):
                        matched = True
                
                else:
                    # Unknown category - accept all
                    matched = True
                
                if matched:
                    filtered_markets.append(market)
            
            print(f"  → {len(filtered_markets)} markets match '{category}'")
            
            # If nothing matched, use all valid markets
            if not filtered_markets:
                print(f"  → Using all valid markets")
                filtered_markets = valid_markets
            
            # Sort by volume if available
            def sort_key(m):
                vol = m.get('volume_24h', 0) or 0
                oi = m.get('open_interest', 0) or 0
                return vol + oi * 0.1
            
            filtered_markets.sort(key=sort_key, reverse=True)
            
            result = filtered_markets[:limit]
            
            if result:
                print(f"  ✓ Returning {len(result)} markets")
                sample = result[0]
                print(f"  Sample: {sample.get('ticker', 'N/A')[:50]}")
            
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_market_details(self, market_ticker: str) -> Optional[Dict]:
        """Get market details"""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}",
                timeout=10
            )
            
            if response.ok:
                return response.json().get("market")
            else:
                print(f"Error fetching market details: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching market details: {e}")
            return None
    
    def get_market_history(self, market_ticker: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}/history",
                params={
                    "min_ts": int(start_time.timestamp()),
                    "max_ts": int(end_time.timestamp())
                },
                timeout=10
            )
            
            if response.status_code == 404:
                return pd.DataFrame()
            
            if response.ok:
                history = response.json().get("history", [])
                if history:
                    df = pd.DataFrame(history)
                    df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
                    return df
            
            return pd.DataFrame()
            
        except:
            return pd.DataFrame()
    
    def extract_market_features(self, market_ticker: str) -> Dict:
        """Extract features with robust defaults"""
        
        # Start with safe defaults
        features = {
            'current_price': 0.5,
            'volume_24h': 100,
            'open_interest': 1000,
            'bid_ask_spread': 0.05,
            'days_to_expiration': 30,
            'price_change_7d': 0,
            'volatility': 0.05,
            'volume_trend': 0
        }
        
        try:
            market = self.get_market_details(market_ticker)
            
            if market:
                # Update with actual data if available
                last_price = market.get('last_price')
                if last_price and last_price > 0:
                    features['current_price'] = last_price / 100
                
                volume_24h = market.get('volume_24h')
                if volume_24h is not None and volume_24h > 0:
                    features['volume_24h'] = volume_24h
                
                open_interest = market.get('open_interest')
                if open_interest is not None and open_interest > 0:
                    features['open_interest'] = open_interest
                
                # Days to expiration
                if market.get('close_time'):
                    try:
                        close_time_str = market['close_time'].replace('Z', '+00:00')
                        close_time = datetime.fromisoformat(close_time_str)
                        now = datetime.now(timezone.utc)
                        days = (close_time - now).days
                        features['days_to_expiration'] = max(days, 1)
                    except:
                        pass
                
                # Try to get historical data
                history_df = self.get_market_history(market_ticker, days=30)
                
                if not history_df.empty and len(history_df) >= 2:
                    try:
                        if len(history_df) >= 7:
                            recent = history_df['price'].iloc[-1]
                            week_ago = history_df['price'].iloc[-7]
                            if week_ago != 0:
                                features['price_change_7d'] = (recent - week_ago) / week_ago
                        
                        returns = history_df['price'].pct_change().dropna()
                        if len(returns) > 0:
                            features['volatility'] = returns.std()
                    except:
                        pass
        
        except Exception as e:
            print(f"Warning: Could not extract all features: {e}")
        
        return features
    
    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """Search markets"""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets",
                params={"limit": limit, "event_ticker": query},
                timeout=10
            )
            
            if response.ok:
                return response.json().get("markets", [])
            else:
                return []
                
        except Exception as e:
            print(f"Error searching markets: {e}")
            return []


if __name__ == "__main__":
    print("Testing Ultra-Simple Kalshi Service...")
    print("="*70)
    
    service = KalshiService()
    
    print("\nTest 1: Fetch any markets")
    markets = service._fetch_markets_simple(limit=10)
    print(f"Got {len(markets)} markets\n")
    
    if markets:
        print("Test 2: Get by category")
        sports_markets = service.get_markets_by_category("Sports", limit=5)
        print(f"\nGot {len(sports_markets)} sports markets")
