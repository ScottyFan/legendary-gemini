"""
Kalshi API Service
Handles all interactions with Kalshi API for market data retrieval
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd


class KalshiService:
    """Service for interacting with Kalshi API"""
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, email: str = None, password: str = None):
        """
        Initialize Kalshi service with optional authentication
        
        Args:
            email: Kalshi account email (optional, for authenticated endpoints)
            password: Kalshi account password (optional)
        """
        self.session = requests.Session()
        self.token = None
        
        if email and password:
            self.login(email, password)
    
    def login(self, email: str, password: str) -> bool:
        """
        Authenticate with Kalshi API
        
        Args:
            email: Account email
            password: Account password
            
        Returns:
            True if login successful
        """
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
    
    def get_markets_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """
        Fetch markets filtered by category
        
        Args:
            category: Category name (e.g., 'Politics', 'Sports', 'Culture', 'Crypto', 'Trump')
            limit: Maximum number of markets to return
            
        Returns:
            List of market dictionaries
        """
        try:
            params = {
                "limit": limit,
                "status": "open"
            }
            
            # Add category filter if it's a standard category
            if category.lower() != 'trump':
                params["series_ticker"] = category.upper()
            
            response = self.session.get(
                f"{self.BASE_URL}/markets",
                params=params
            )
            response.raise_for_status()
            
            markets = response.json().get("markets", [])
            
            # For 'Trump' category, filter by ticker or title containing Trump
            if category.lower() == 'trump':
                markets = [
                    m for m in markets 
                    if 'trump' in m.get('title', '').lower() or 
                       'trump' in m.get('ticker', '').lower()
                ]
            
            return markets
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []
    
    def get_market_details(self, market_ticker: str) -> Optional[Dict]:
        """
        Get detailed information about a specific market
        
        Args:
            market_ticker: Market ticker symbol
            
        Returns:
            Market details dictionary or None
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}"
            )
            response.raise_for_status()
            return response.json().get("market")
        except Exception as e:
            print(f"Error fetching market details: {e}")
            return None
    
    def get_market_history(self, market_ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical price and volume data for a market
        
        Args:
            market_ticker: Market ticker symbol
            days: Number of days of history to fetch
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Get orderbook history or trades
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}/history",
                params={
                    "min_ts": int(start_time.timestamp()),
                    "max_ts": int(end_time.timestamp())
                }
            )
            response.raise_for_status()
            
            history = response.json().get("history", [])
            
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
            return df
        except Exception as e:
            print(f"Error fetching market history: {e}")
            return pd.DataFrame()
    
    def extract_market_features(self, market_ticker: str) -> Dict:
        """
        Extract meaningful features from a market for ML prediction
        
        Args:
            market_ticker: Market ticker symbol
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Get current market details
        market = self.get_market_details(market_ticker)
        if not market:
            return features
        
        # Get historical data
        history_df = self.get_market_history(market_ticker, days=30)
        
        # Current state features
        features['current_price'] = market.get('last_price', 0) / 100  # Convert cents to dollars
        features['volume_24h'] = market.get('volume_24h', 0)
        features['open_interest'] = market.get('open_interest', 0)
        
        # Calculate bid-ask spread if available
        yes_bid = market.get('yes_bid', 0)
        yes_ask = market.get('yes_ask', 0)
        features['bid_ask_spread'] = (yes_ask - yes_bid) / 100 if yes_ask and yes_bid else 0
        
        # Time-based features
        if market.get('close_time'):
            close_time = datetime.fromisoformat(market['close_time'].replace('Z', '+00:00'))
            features['days_to_expiration'] = (close_time - datetime.now()).days
        else:
            features['days_to_expiration'] = 0
        
        # Historical features (if we have history)
        if not history_df.empty:
            # Price momentum
            if len(history_df) >= 7:
                features['price_change_7d'] = (
                    history_df['price'].iloc[-1] - history_df['price'].iloc[-7]
                ) / history_df['price'].iloc[-7] if history_df['price'].iloc[-7] != 0 else 0
            
            # Volatility (standard deviation of returns)
            if len(history_df) >= 2:
                returns = history_df['price'].pct_change().dropna()
                features['volatility'] = returns.std() if len(returns) > 0 else 0
            
            # Volume trend
            if len(history_df) >= 7:
                recent_volume = history_df['volume'].iloc[-7:].mean()
                older_volume = history_df['volume'].iloc[:-7].mean() if len(history_df) > 14 else recent_volume
                features['volume_trend'] = (
                    (recent_volume - older_volume) / older_volume 
                    if older_volume > 0 else 0
                )
        
        # Fill missing features with defaults
        default_features = {
            'price_change_7d': 0,
            'volatility': 0,
            'volume_trend': 0
        }
        for key, value in default_features.items():
            if key not in features:
                features[key] = value
        
        return features
    
    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for markets by keyword
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching markets
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets",
                params={
                    "limit": limit,
                    "event_ticker": query
                }
            )
            response.raise_for_status()
            return response.json().get("markets", [])
        except Exception as e:
            print(f"Error searching markets: {e}")
            return []


# Example usage
if __name__ == "__main__":
    service = KalshiService()
    
    # Get Trump-related markets
    trump_markets = service.get_markets_by_category("Trump", limit=10)
    print(f"Found {len(trump_markets)} Trump-related markets")
    
    if trump_markets:
        # Get features for first market
        ticker = trump_markets[0]['ticker']
        features = service.extract_market_features(ticker)
        print(f"\nFeatures for {ticker}:")
        for key, value in features.items():
            print(f"  {key}: {value}")
