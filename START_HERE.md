# 🚀 START HERE

Welcome to the Kalshi Market Predictor! This guide will get you up and running in **5 minutes**.

## What You Have

A complete full-stack AI-powered app with:
- ✅ React frontend with beautiful UI
- ✅ Python backend with ML predictions
- ✅ Kalshi market data integration
- ✅ News sentiment analysis
- ✅ No API keys needed to start!

## Quick Start (Choose One)

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh
```

Then follow the instructions to start backend and frontend.

### Option 2: Manual Setup

#### Step 1: Backend (2 minutes)

```bash
cd backend

# Install Python packages
pip install requests beautifulsoup4 pandas numpy scikit-learn joblib fastapi uvicorn pydantic

# Test it works
python simple_test.py

# Start backend (KEEP THIS RUNNING)
uvicorn api_server:app --reload --port 8000
```

#### Step 2: Frontend (2 minutes)

**Open a NEW terminal:**

```bash
cd frontend

# Install Node packages
npm install

# Start frontend
npm start
```

Browser will open automatically at `http://localhost:3000` 🎉

## What to Expect

1. **Category Selection**: Choose Politics, Sports, Culture, Crypto, or Trump
2. **Market Browser**: See live Kalshi markets with prices
3. **AI Prediction**: Click any market to get ML-powered recommendation
4. **Results**: View Buy/Sell/Hold with confidence scores

## Folder Structure

```
kalshi-predictor-app/
├── backend/          → Python API (FastAPI + ML)
├── frontend/         → React UI (beautiful dark theme)
├── setup.sh          → Automatic setup script
└── README.md         → Full documentation
```

## Common Issues

**"Module not found" error**
```bash
cd backend
pip install -r requirements_minimal.txt
```

**"Port 8000 already in use"**
```bash
# Kill existing process or use different port
uvicorn api_server:app --reload --port 8001
# Then update frontend/src/App.jsx: API_BASE_URL = 'http://localhost:8001'
```

**"npm install" fails**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Frontend shows "API Error"**
- Make sure backend is running on port 8000
- Check terminal running backend for errors
- Try: `curl http://localhost:8000/api/categories`

## Features Explained

### ML Model
- Trains on synthetic data initially
- Uses 17+ features (market + news)
- Random Forest classifier
- 5 output classes: Strong Buy → Strong Sell

### News Analysis
- Scrapes CNN, Reuters, BBC
- Extracts sentiment scores
- Analyzes article count and recency
- No API keys needed!

### Market Data
- Live from Kalshi public API
- No authentication required
- Real prices and volumes
- Historical trends

## Next Steps

1. ✅ Get it running (you're here!)
2. 📊 Try different categories
3. 🎯 Get some predictions
4. 🔧 Customize (see README.md)
5. 🚀 Deploy (optional)

## Optional Enhancements

### Add Gemini AI (Better Analysis)
1. Get key: https://aistudio.google.com/app/apikey
2. Edit `backend/api_server.py`: `use_gemini=True`
3. `export GEMINI_API_KEY="your-key"`
4. Restart backend

### Train on Real Data
See `README.md` section on "Model Training"

### Customize UI
Edit `frontend/src/App.jsx`

## File Guide

**Backend Files:**
- `api_server.py` - Main API server (START HERE)
- `integration_service.py` - Orchestrates everything
- `kalshi_service.py` - Fetches market data
- `news_scraper.py` - Scrapes news
- `ml_classifier.py` - ML predictions
- `simple_test.py` - Quick test script

**Frontend Files:**
- `src/App.jsx` - Main React component
- `src/index.js` - React entry
- `src/index.css` - Tailwind styles
- `package.json` - Dependencies

## Need Help?

1. Check backend terminal for errors
2. Check browser console (F12)
3. Read full `README.md`
4. Backend logs show what's happening

## Testing

**Quick backend test:**
```bash
cd backend
python simple_test.py
```

**Check API:**
```bash
curl http://localhost:8000/
curl http://localhost:8000/api/categories
```

**Check frontend:**
Visit `http://localhost:3000` in browser

## You're All Set! 🎉

Open `http://localhost:3000` and start exploring markets!

---

**Pro Tips:**
- Backend terminal shows all API requests
- React auto-reloads when you edit code
- Model predictions improve with real training data
- Dark mode UI looks best in dim lighting 😎

**Happy predicting!** 📈📉
