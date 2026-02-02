"""
FastAPI Backend Service
Provides REST API endpoints for the React frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from integration_service import MarketPredictionService

# Initialize FastAPI app
app = FastAPI(
    title="Kalshi Market Prediction API",
    description="AI-powered market predictions using ML and sentiment analysis",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = MarketPredictionService(
    kalshi_email=os.getenv("KALSHI_EMAIL"),
    kalshi_password=os.getenv("KALSHI_PASSWORD"),
    gemini_api_key=os.getenv("GEMINI_API_KEY")
)

# Request/Response Models
class PredictionRequest(BaseModel):
    market_ticker: str
    category: Optional[str] = None

class MarketSummary(BaseModel):
    ticker: str
    title: str
    price: float
    volume: int
    close_time: Optional[str] = None

class PredictionResponse(BaseModel):
    ticker: str
    market: Dict
    prediction: Dict
    analysis: Dict
    news: Dict
    context: Optional[Dict] = None


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Kalshi Market Prediction API",
        "version": "1.0.0"
    }

@app.get("/api/categories", response_model=List[str])
async def get_categories():
    """Get available market categories"""
    return prediction_service.get_categories()

@app.get("/api/markets/{category}", response_model=List[MarketSummary])
async def get_markets(category: str, limit: int = 20):
    """
    Get markets for a specific category
    
    Args:
        category: Category name (Politics, Sports, Culture, Crypto, Trump)
        limit: Maximum number of markets to return
    """
    try:
        markets = prediction_service.get_markets_by_category(category, limit)
        return markets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def get_prediction(request: PredictionRequest):
    """
    Get AI prediction for a specific market
    
    Args:
        request: PredictionRequest with market_ticker and optional category
    """
    try:
        prediction = await prediction_service.get_full_prediction(
            request.market_ticker,
            request.category
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=404, detail=prediction['error'])
        
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/{ticker}")
async def get_market_details(ticker: str):
    """Get detailed information about a specific market"""
    try:
        details = prediction_service.kalshi.get_market_details(ticker)
        
        if not details:
            raise HTTPException(status_code=404, detail="Market not found")
        
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news/{query}")
async def search_news(query: str, max_articles: int = 10):
    """
    Search for news articles related to a query
    
    Args:
        query: Search query
        max_articles: Maximum articles to return
    """
    try:
        articles = prediction_service.scraper.search_news(query, max_articles=max_articles)
        return {
            'query': query,
            'count': len(articles),
            'articles': articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_model():
    """
    Train the ML model (admin endpoint)
    Note: In production, this should be protected with authentication
    """
    try:
        metrics = prediction_service.train_model()
        return {
            'status': 'success',
            'metrics': metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/status")
async def get_model_status():
    """Get information about the current ML model"""
    return {
        'is_trained': prediction_service.classifier.is_trained,
        'model_type': prediction_service.classifier.model_type,
        'feature_count': len(prediction_service.classifier.feature_names),
        'features': prediction_service.classifier.feature_names
    }


# Run with: uvicorn api_server:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
