import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, AlertCircle, ChevronRight, Calendar, DollarSign, Activity } from 'lucide-react';

// API configuration
const API_BASE_URL = 'http://localhost:8000';

// API service
const api = {
  async getCategories() {
    const res = await fetch(`${API_BASE_URL}/api/categories`);
    if (!res.ok) throw new Error('Failed to fetch categories');
    return res.json();
  },
  
  async getMarkets(category) {
    const res = await fetch(`${API_BASE_URL}/api/markets/${category}`);
    if (!res.ok) throw new Error('Failed to fetch markets');
    return res.json();
  },
  
  async getPrediction(ticker, category) {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ market_ticker: ticker, category: category })
    });
    if (!res.ok) throw new Error('Failed to get prediction');
    return res.json();
  }
};

function App() {
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [markets, setMarkets] = useState([]);
  const [selectedMarket, setSelectedMarket] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadCategories();
  }, []);

  const loadCategories = async () => {
    try {
      const cats = await api.getCategories();
      setCategories(cats);
    } catch (err) {
      setError('Failed to load categories. Is the API server running?');
      console.error(err);
    }
  };

  useEffect(() => {
    if (selectedCategory) {
      loadMarkets();
    }
  }, [selectedCategory]);

  const loadMarkets = async () => {
    setMarkets([]);
    setSelectedMarket(null);
    setPrediction(null);
    setError(null);
    setLoading(true);
    
    try {
      const mkts = await api.getMarkets(selectedCategory);
      setMarkets(mkts);
    } catch (err) {
      setError('Failed to load markets. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleMarketSelect = async (market) => {
    setSelectedMarket(market);
    setPrediction(null);
    setError(null);
    setLoading(true);
    
    try {
      const pred = await api.getPrediction(market.ticker, selectedCategory);
      setPrediction(pred);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (pred) => {
    const colors = {
      strong_buy: 'from-emerald-500 to-teal-600',
      buy: 'from-green-500 to-emerald-600',
      hold: 'from-amber-500 to-orange-600',
      sell: 'from-red-500 to-rose-600',
      strong_sell: 'from-rose-600 to-red-700'
    };
    return colors[pred] || 'from-gray-500 to-gray-600';
  };

  const getPredictionIcon = (pred) => {
    if (pred.includes('buy')) return <TrendingUp className="w-8 h-8" />;
    if (pred.includes('sell')) return <TrendingDown className="w-8 h-8" />;
    return <Minus className="w-8 h-8" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900">
      {/* Background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40rem] h-[40rem] bg-fuchsia-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-white/10 backdrop-blur-xl bg-white/5">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Activity className="w-10 h-10 text-indigo-400" strokeWidth={1.5} />
                <div>
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-200 to-violet-200 bg-clip-text text-transparent">
                    Kalshi Market Intelligence
                  </h1>
                  <p className="text-sm text-indigo-300/60 mt-1">AI-Powered Market Predictions</p>
                </div>
              </div>
              {error && (
                <div className="flex items-center gap-2 px-4 py-2 bg-red-500/20 border border-red-500/50 rounded-lg">
                  <AlertCircle className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-200">API Error</span>
                </div>
              )}
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-12">
          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
              <p className="text-red-200">{error}</p>
              <p className="text-sm text-red-300/70 mt-2">
                Make sure the backend is running: <code className="bg-black/30 px-2 py-1 rounded">uvicorn api_server:app --reload</code>
              </p>
            </div>
          )}

          {/* Category Selection */}
          {!selectedCategory && (
            <div className="space-y-8 animate-fade-in">
              <div className="text-center mb-12">
                <h2 className="text-5xl font-bold text-white mb-4">Choose Your Category</h2>
                <p className="text-xl text-indigo-200/70">Explore prediction markets across different sectors</p>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {categories.map((category, idx) => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-white/10 to-white/5 border border-white/20 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-105 hover:border-indigo-400/50 hover:shadow-2xl hover:shadow-indigo-500/20"
                    style={{ animationDelay: `${idx * 100}ms` }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/0 to-violet-500/0 group-hover:from-indigo-500/10 group-hover:to-violet-500/10 transition-all duration-300"></div>
                    <div className="relative z-10">
                      <div className="text-4xl mb-4 opacity-80 group-hover:opacity-100 transition-opacity">
                        {category === 'Politics' && '🏛️'}
                        {category === 'Sports' && '⚽'}
                        {category === 'Culture' && '🎭'}
                        {category === 'Crypto' && '₿'}
                        {category === 'Trump' && '🇺🇸'}
                      </div>
                      <h3 className="text-xl font-bold text-white group-hover:text-indigo-200 transition-colors">
                        {category}
                      </h3>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Market Browser */}
          {selectedCategory && !selectedMarket && (
            <div className="space-y-6 animate-fade-in">
              <button
                onClick={() => setSelectedCategory(null)}
                className="text-indigo-300 hover:text-indigo-200 flex items-center gap-2 mb-6 transition-colors"
              >
                ← Back to Categories
              </button>
              
              <div className="mb-8">
                <h2 className="text-4xl font-bold text-white mb-2">{selectedCategory} Markets</h2>
                <p className="text-indigo-200/70">Select a market to view AI predictions</p>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-20">
                  <div className="relative">
                    <div className="w-16 h-16 border-4 border-indigo-400/30 border-t-indigo-400 rounded-full animate-spin"></div>
                    <p className="text-indigo-300 mt-6 text-center">Loading markets...</p>
                  </div>
                </div>
              ) : markets.length === 0 ? (
                <div className="text-center py-20">
                  <p className="text-indigo-300 text-lg">No markets found for this category</p>
                  <p className="text-indigo-400/60 text-sm mt-2">Try another category or check back later</p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {markets.map((market, idx) => (
                    <button
                      key={market.ticker}
                      onClick={() => handleMarketSelect(market)}
                      className="group text-left rounded-2xl bg-gradient-to-br from-white/10 to-white/5 border border-white/20 p-6 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-indigo-400/50 hover:shadow-xl hover:shadow-indigo-500/10"
                      style={{ animationDelay: `${idx * 50}ms` }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="text-xs font-mono text-indigo-400 mb-2">{market.ticker}</div>
                          <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-indigo-200 transition-colors">
                            {market.title}
                          </h3>
                          <div className="flex gap-6 text-sm">
                            <div className="flex items-center gap-2 text-indigo-200/70">
                              <DollarSign className="w-4 h-4" />
                              <span>${market.price.toFixed(2)}</span>
                            </div>
                            <div className="flex items-center gap-2 text-indigo-200/70">
                              <Activity className="w-4 h-4" />
                              <span>{(market.volume / 1000).toFixed(0)}K volume</span>
                            </div>
                          </div>
                        </div>
                        <ChevronRight className="w-6 h-6 text-indigo-400 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Prediction Display */}
          {selectedMarket && (
            <div className="space-y-6 animate-fade-in">
              <button
                onClick={() => setSelectedMarket(null)}
                className="text-indigo-300 hover:text-indigo-200 flex items-center gap-2 mb-6 transition-colors"
              >
                ← Back to Markets
              </button>

              <div className="rounded-3xl bg-gradient-to-br from-white/10 to-white/5 border border-white/20 p-8 backdrop-blur-sm">
                <div className="text-xs font-mono text-indigo-400 mb-2">{selectedMarket.ticker}</div>
                <h2 className="text-3xl font-bold text-white mb-6">{selectedMarket.title}</h2>

                {loading ? (
                  <div className="flex items-center justify-center py-20">
                    <div className="relative">
                      <div className="w-16 h-16 border-4 border-indigo-400/30 border-t-indigo-400 rounded-full animate-spin"></div>
                      <p className="text-indigo-300 mt-6 text-center">Analyzing market data...</p>
                    </div>
                  </div>
                ) : prediction ? (
                  <div className="space-y-6">
                    {/* Main Prediction */}
                    <div className={`rounded-2xl bg-gradient-to-br ${getPredictionColor(prediction.prediction.recommendation)} p-8 text-white shadow-2xl`}>
                      <div className="flex items-center gap-4 mb-4">
                        {getPredictionIcon(prediction.prediction.recommendation)}
                        <div>
                          <div className="text-sm opacity-90 mb-1">AI Recommendation</div>
                          <div className="text-3xl font-bold uppercase tracking-wide">
                            {prediction.prediction.recommendation.replace('_', ' ')}
                          </div>
                        </div>
                      </div>
                      <div className="mt-4 pt-4 border-t border-white/20">
                        <div className="text-sm opacity-90 mb-2">Confidence Level</div>
                        <div className="flex items-center gap-4">
                          <div className="flex-1 h-3 bg-white/20 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-white rounded-full transition-all duration-1000"
                              style={{ width: `${prediction.prediction.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xl font-bold">{(prediction.prediction.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Analysis Grid */}
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="rounded-xl bg-white/5 border border-white/10 p-6">
                        <h3 className="text-sm font-semibold text-indigo-300 mb-4">Market Indicators</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Current Price</span>
                            <span className="text-white font-semibold">
                              ${prediction.market.price.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">24h Volume</span>
                            <span className="text-white font-semibold">
                              {(prediction.market.volume_24h / 1000).toFixed(0)}K
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Price Change (7d)</span>
                            <span className="text-white font-semibold">
                              {prediction.analysis.market_features.price_change_7d > 0 ? '+' : ''}
                              {(prediction.analysis.market_features.price_change_7d * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Volatility</span>
                            <span className="text-white font-semibold">
                              {prediction.analysis.market_features.volatility.toFixed(3)}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-xl bg-white/5 border border-white/10 p-6">
                        <h3 className="text-sm font-semibold text-indigo-300 mb-4">News Sentiment</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Articles Analyzed</span>
                            <span className="text-white font-semibold">
                              {prediction.news.article_count}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Avg Sentiment</span>
                            <span className="text-white font-semibold">
                              {prediction.analysis.news_features.avg_sentiment > 0 ? '+' : ''}
                              {prediction.analysis.news_features.avg_sentiment.toFixed(2)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Positive Ratio</span>
                            <span className="text-white font-semibold">
                              {(prediction.analysis.news_features.positive_ratio * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-indigo-200/70 text-sm">Source Diversity</span>
                            <span className="text-white font-semibold">
                              {(prediction.analysis.news_features.source_diversity * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Probabilities */}
                    <div className="rounded-xl bg-white/5 border border-white/10 p-6">
                      <h3 className="text-sm font-semibold text-indigo-300 mb-4">Class Probabilities</h3>
                      <div className="space-y-2">
                        {Object.entries(prediction.prediction.probabilities)
                          .sort(([,a], [,b]) => b - a)
                          .map(([cls, prob]) => (
                            <div key={cls} className="flex items-center gap-3">
                              <span className="text-sm text-indigo-200/80 w-24 capitalize">
                                {cls.replace('_', ' ')}
                              </span>
                              <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-indigo-400 rounded-full transition-all duration-1000"
                                  style={{ width: `${prob * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm text-white font-semibold w-12 text-right">
                                {(prob * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>

                    {/* AI Explanation */}
                    <div className="rounded-xl bg-gradient-to-br from-indigo-500/10 to-violet-500/10 border border-indigo-400/30 p-6">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-indigo-400 mt-0.5 flex-shrink-0" />
                        <div>
                          <h3 className="text-sm font-semibold text-indigo-300 mb-2">Analysis</h3>
                          <p className="text-indigo-100/80 text-sm leading-relaxed">
                            {prediction.prediction.explanation}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Top Features */}
                    {prediction.analysis.top_features && prediction.analysis.top_features.length > 0 && (
                      <div className="rounded-xl bg-white/5 border border-white/10 p-6">
                        <h3 className="text-sm font-semibold text-indigo-300 mb-4">Top Contributing Features</h3>
                        <div className="space-y-2">
                          {prediction.analysis.top_features.slice(0, 5).map((feature, idx) => (
                            <div key={idx} className="flex justify-between items-center text-sm">
                              <span className="text-indigo-200/70">{feature.name}</span>
                              <span className="text-white font-mono">{feature.value.toFixed(3)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </main>
      </div>

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.6s ease-out forwards;
        }
      `}</style>
    </div>
  );
}

export default App;
