# Kalshi Market Prediction App

🚀 **Complete full-stack AI-powered prediction platform** that combines Kalshi market data, news sentiment analysis, and machine learning to provide intelligent market recommendations.

## 🎯 Features

- **Smart Category Navigation**: Browse markets across Politics, Sports, Culture, Crypto, and Trump
- **Real-time Market Data**: Live data from Kalshi API
- **News Sentiment Analysis**: Web scraping from CNN, Reuters, BBC
- **ML Predictions**: Random Forest classifier with buy/sell/hold recommendations
- **Beautiful UI**: Modern React interface with smooth animations and dark gradients
- **No Authentication Needed**: Works out of the box with public Kalshi data

## 📁 Project Structure

```
kalshi-predictor-app/
├── backend/                    # Python FastAPI backend
│   ├── kalshi_service.py      # Kalshi API integration
│   ├── news_scraper.py        # Web scraping & sentiment
│   ├── gemini_analyzer.py     # Optional Gemini AI
│   ├── ml_classifier.py       # ML prediction model
│   ├── integration_service.py # Main orchestration
│   ├── api_server.py          # FastAPI REST API
│   ├── simple_test.py         # Quick test script
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.jsx            # Main React app
│   │   ├── index.js           # React entry point
│   │   └── index.css          # Tailwind styles
│   ├── public/
│   │   └── index.html         # HTML template
│   ├── package.json           # Node dependencies
│   ├── tailwind.config.js     # Tailwind config
│   └── postcss.config.js      # PostCSS config
│
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### 1. Backend Setup (2 minutes)

```bash
cd backend

# Install Python dependencies
pip install requests beautifulsoup4 pandas numpy scikit-learn joblib fastapi uvicorn pydantic

# Test it works
python simple_test.py

# Start the API server (keep this running)
uvicorn api_server:app --reload --port 8000
```

The backend will be running at `http://localhost:8000`

### 2. Frontend Setup (2 minutes)

Open a new terminal:

```bash
cd frontend

# Install Node dependencies
npm install

# Start React dev server
npm start
```

The app will open at `http://localhost:3000` 🎉

## 📖 Usage

1. **Select a Category** (Politics, Sports, Culture, Crypto, Trump)
2. **Browse Markets** (see current prices and volumes)
3. **Get AI Prediction** (click any market)
4. **View Results**:
   - Buy/Sell/Hold recommendation
   - Confidence score
   - Market analysis
   - News sentiment
   - Feature importance

## 🛠️ Technical Details

### Backend Stack
- **FastAPI**: REST API framework
- **scikit-learn**: ML classification
- **BeautifulSoup**: Web scraping
- **Pandas/NumPy**: Data processing

### Frontend Stack
- **React 18**: UI framework
- **Tailwind CSS**: Styling
- **Lucide React**: Icons

### ML Model
- **Algorithm**: Random Forest Classifier
- **Features**: 17-20 combined features
  - Market: price, volume, volatility, momentum
  - News: sentiment, article count, source diversity
  - Interactions: combined signals
- **Classes**: Strong Buy, Buy, Hold, Sell, Strong Sell

### API Endpoints

```
GET  /api/categories           # List categories
GET  /api/markets/{category}   # Get markets
POST /api/predict              # Get prediction
GET  /api/market/{ticker}      # Market details
GET  /api/news/{query}         # Search news
POST /api/train                # Train model
GET  /api/model/status         # Model info
```

## 🎨 UI Features

- Dark gradient background (slate/indigo/violet)
- Smooth animations and transitions
- Responsive design
- Real-time loading states
- Error handling with helpful messages
- Color-coded predictions
- Interactive probability bars

## 🔧 Configuration

### Backend (`backend/api_server.py`)

```python
# Enable Gemini (optional)
prediction_service = MarketPredictionService(
    use_gemini=True  # Set to False by default
)
```

### Frontend (`frontend/src/App.jsx`)

```javascript
// Change API URL if backend runs on different port
const API_BASE_URL = 'http://localhost:8000';
```

## 📊 How It Works

1. **User selects category** → Fetches markets from Kalshi
2. **User picks market** → Prediction pipeline starts:
   - Fetches market features (price, volume, volatility)
   - Scrapes news articles (CNN, Reuters, BBC)
   - Analyzes sentiment (basic or Gemini-enhanced)
   - Combines features into vector
   - ML model predicts outcome
   - Returns recommendation with confidence

## 🧪 Testing

### Backend Test
```bash
cd backend
python simple_test.py
```

### Frontend Test
Visit `http://localhost:3000` after starting both servers

### API Test
```bash
# Health check
curl http://localhost:8000/

# Get categories
curl http://localhost:8000/api/categories

# Get markets
curl http://localhost:8000/api/markets/Trump
```

## 🐛 Troubleshooting

**Backend won't start**
- Check Python version: `python --version` (need 3.9+)
- Install dependencies: `pip install -r requirements.txt`
- Port 8000 taken: Change port in `uvicorn` command

**Frontend won't start**
- Check Node version: `node --version` (need 16+)
- Delete node_modules: `rm -rf node_modules && npm install`
- Port 3000 taken: React will auto-prompt for different port

**API errors in frontend**
- Ensure backend is running on port 8000
- Check CORS settings in `api_server.py`
- Check browser console for details

**No markets found**
- Kalshi API might be rate-limiting
- Wait a few seconds and try again
- Check network connectivity

**Prediction fails**
- Model might not be trained
- Run `python simple_test.py` to train
- Check backend logs for errors

## 🔐 Optional: Gemini Integration

For enhanced analysis (completely optional):

1. Get API key: https://aistudio.google.com/app/apikey
2. In `backend/api_server.py`:
   ```python
   prediction_service = MarketPredictionService(use_gemini=True)
   ```
3. Set environment: `export GEMINI_API_KEY="your-key"`
4. Restart backend

## 📦 Deployment

### Backend
- Docker: Create Dockerfile in `backend/`
- Heroku: `heroku create` + `git push heroku main`
- AWS Lambda: Use Mangum adapter
- Railway: Connect repo

### Frontend
```bash
cd frontend
npm run build
```
Deploy `build/` folder to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add more news sources
- Improve ML model
- Enhance UI/UX
- Add new features

## 📝 License

MIT License - Use freely!

## 🙏 Credits

- Kalshi API for market data
- News sources (CNN, Reuters, BBC)
- Google Gemini for AI analysis (optional)

## 📧 Support

For issues:
1. Check this README
2. Review backend/frontend logs
3. Open a GitHub issue

---

**Built with ❤️ for better market predictions**

🚀 Happy trading!
