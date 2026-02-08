# 🔮 Kalshi Market Predictor

An AI-powered analysis engine designed to predict outcomes on Kalshi prediction markets using a blend of **machine learning** and **real-time sentiment analysis**.

---

## 🎯 Overview

The Predictor evaluates specific market tickers by aggregating data across two primary vectors:
* **Market Dynamics:** Real-time price action, volume trends, and volatility.
* **External Sentiment:** Live news scraping from Google News to gauge public and institutional bias.

**The Result:** A clear "YES/NO" signal backed by a confidence percentage and a breakdown of contributing factors.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.8+ installed. Clone the repository and install the necessary dependencies:

```bash
pip install -r requirements.txt
```

**Required Packages:**

* `fastapi`, `uvicorn` (Backend API)
* `scikit-learn`, `pandas`, `numpy` (Machine Learning)
* `requests`, `feedparser`, `beautifulsoup4` (Data Scraping)

### 2. Train the Model

Before running predictions, you must train the Random Forest classifier on historical or current snapshots:

```bash
python ml_classifier.py
```

*This creates `kalshi_final_model.pkl` (Approx. 5-minute runtime).*

### 3. Start the API

Launch the FastAPI backend server:

```bash
python api_server.py
```

*The server will be live at:* **http://localhost:8000**

---

## 🔌 API Endpoints

### `POST /predict`

Get an AI-driven prediction for a specific market ticker.

**Request Body:**

```json
{
  "ticker": "KXBTC-26DEC31-B100K"
}

```

**Sample Response:**

```json
{
  "prediction": "YES",
  "confidence": 0.73,
  "explanation": "The model is confident (73%) that this market will resolve to YES.",
  "key_factors": [
    "Market price strongly favors YES",
    "Positive news sentiment detected"
  ]
}

```

### `GET /markets/search`

Search for active Kalshi markets by keyword.

* **Query:** `?query=Bitcoin&limit=10`

---

## 🧠 Technical Architecture

### How It Works

1. **Data Ingestion:** Pulls order book data (price, volume, bid-ask spread) from the Kalshi Public API.
2. **Sentiment Engine:** Scrapes Google News RSS for headlines and snippets related to the ticker topic.
3. **Feature Engineering:** Extracts **23 distinct features** (12 market-based, 11 sentiment-based).
4. **Inference:** A **Random Forest Classifier** processes the features to output a probability score.

### Project Structure

```text
kalshi-predictor/
├── api.py                  # FastAPI backend implementation
├── ml_classifier.py          # ML training script & logic
├── kalshi_service.py       # Kalshi API integration
├── news_scraper.py         # Google News sentiment engine
├── frontend.html           # Minimalist Web UI
└── kalshi_final_model.pkl  # Trained model binary (git-ignored)

```

---

## 🔧 Troubleshooting

* **"Model not loaded":** You must run `train_final.py` at least once to generate the `.pkl` file.
* **"Market not found":** Ensure the ticker format matches Kalshi's official naming convention (e.g., `KXBTC-26DEC31-B100K`).
* **"No articles found":** Occurs for very niche markets. The model will default to market-data features only.
* **Port 8000 in use:** Change the port in `api.py` using `uvicorn.run(app, port=8001)`.

---

## 📝 License

Distributed under the MIT License. Feel free to use and modify for personal or commercial use.

**Built with ❤️ using FastAPI, scikit-learn, and BeautifulSoup**

```

Would you like me to help you write the code for the `requirements.txt` file or the `train_final.py` logic next?

```