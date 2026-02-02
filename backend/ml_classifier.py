"""
ML Classifier for Market Predictions
Combines market data and news sentiment to predict market movements
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Optional
import json


class MarketClassifier:
    """ML classifier for predicting market performance"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize classifier
        
        Args:
            model_type: 'random_forest' or 'gradient_boost'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, market_features: Dict, news_features: Dict, 
                         gemini_features: Dict = None) -> np.ndarray:
        """
        Prepare feature vector from market and news data
        
        Args:
            market_features: Dictionary of market features
            news_features: Dictionary of news features
            gemini_features: Optional Gemini analysis features
            
        Returns:
            Feature array
        """
        features = []
        feature_names = []
        
        # Market features
        market_keys = [
            'current_price', 'volume_24h', 'open_interest',
            'bid_ask_spread', 'days_to_expiration',
            'price_change_7d', 'volatility', 'volume_trend'
        ]
        
        for key in market_keys:
            features.append(market_features.get(key, 0))
            feature_names.append(f'market_{key}')
        
        # News features
        news_keys = [
            'article_count', 'avg_sentiment', 'sentiment_std',
            'positive_ratio', 'negative_ratio', 'source_diversity', 'recency_score'
        ]
        
        for key in news_keys:
            features.append(news_features.get(key, 0))
            feature_names.append(f'news_{key}')
        
        # Gemini features (if available)
        if gemini_features:
            gemini_keys = ['sentiment_score', 'confidence', 'event_significance']
            for key in gemini_keys:
                features.append(gemini_features.get(key, 0))
                feature_names.append(f'gemini_{key}')
        
        # Interaction features
        # Market momentum * sentiment
        momentum = market_features.get('price_change_7d', 0)
        sentiment = news_features.get('avg_sentiment', 0)
        features.append(momentum * sentiment)
        feature_names.append('momentum_sentiment_interaction')
        
        # Volume * article count
        volume = market_features.get('volume_24h', 0)
        articles = news_features.get('article_count', 0)
        features.append(np.log1p(volume) * articles)  # Log transform volume
        feature_names.append('volume_news_interaction')
        
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Dict], labels: List[str]) -> Dict:
        """
        Train the classifier on historical data
        
        Args:
            training_data: List of dictionaries with market and news features
            labels: List of outcome labels ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')
            
        Returns:
            Training metrics
        """
        # Prepare feature matrix
        X = []
        for data_point in training_data:
            features = self.prepare_features(
                data_point.get('market_features', {}),
                data_point.get('news_features', {}),
                data_point.get('gemini_features', {})
            )
            X.append(features.flatten())
        
        X = np.array(X)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5
        )
        
        # Predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            feature_importance = {}
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance
        }
    
    def predict(self, market_features: Dict, news_features: Dict, 
                gemini_features: Dict = None) -> Dict:
        """
        Make prediction for a market
        
        Args:
            market_features: Market data features
            news_features: News sentiment features
            gemini_features: Optional Gemini analysis
            
        Returns:
            Prediction with confidence scores
        """
        if not self.is_trained:
            return {
                'prediction': 'hold',
                'confidence': 0,
                'probabilities': {},
                'recommendation': 'Model not trained yet'
            }
        
        # Prepare features
        X = self.prepare_features(market_features, news_features, gemini_features)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get class names
        classes = self.model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Confidence is the probability of the predicted class
        confidence = float(max(probabilities))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction, confidence, prob_dict
        )
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'recommendation': recommendation,
            'top_features': self._get_top_features(X_scaled[0])
        }
    
    def _generate_recommendation(self, prediction: str, confidence: float, 
                                 probabilities: Dict) -> str:
        """Generate human-readable recommendation"""
        
        confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
        
        recommendations = {
            'strong_buy': f"Strong Buy recommendation with {confidence_level} confidence ({confidence:.1%}). "
                         "Market conditions and sentiment are very favorable.",
            'buy': f"Buy recommendation with {confidence_level} confidence ({confidence:.1%}). "
                   "Positive indicators outweigh negative factors.",
            'hold': f"Hold recommendation with {confidence_level} confidence ({confidence:.1%}). "
                    "Mixed signals suggest waiting for clearer direction.",
            'sell': f"Sell recommendation with {confidence_level} confidence ({confidence:.1%}). "
                    "Negative indicators suggest reducing exposure.",
            'strong_sell': f"Strong Sell recommendation with {confidence_level} confidence ({confidence:.1%}). "
                          "Market conditions and sentiment are very unfavorable."
        }
        
        return recommendations.get(prediction, "Unable to generate recommendation")
    
    def _get_top_features(self, feature_vector: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Get most important features for this prediction"""
        
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create list of (feature_name, value, importance) tuples
        features_with_importance = [
            {
                'name': name,
                'value': float(value),
                'importance': float(importance)
            }
            for name, value, importance in zip(
                self.feature_names, feature_vector, importances
            )
        ]
        
        # Sort by importance and take top N
        features_with_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return features_with_importance[:top_n]
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.is_trained = True


# Example usage and synthetic data generation
if __name__ == "__main__":
    # Create synthetic training data for demonstration
    np.random.seed(42)
    
    def generate_synthetic_data(n_samples: int = 200) -> Tuple[List[Dict], List[str]]:
        """Generate synthetic training data"""
        data = []
        labels = []
        
        label_options = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
        
        for _ in range(n_samples):
            # Generate market features
            price_change = np.random.normal(0, 0.1)
            sentiment = np.random.normal(0, 0.3)
            
            # Label based on features (simplified logic)
            if price_change > 0.1 and sentiment > 0.3:
                label = 'strong_buy'
            elif price_change > 0 and sentiment > 0:
                label = 'buy'
            elif price_change < -0.1 and sentiment < -0.3:
                label = 'strong_sell'
            elif price_change < 0 and sentiment < 0:
                label = 'sell'
            else:
                label = 'hold'
            
            data_point = {
                'market_features': {
                    'current_price': np.random.uniform(0.3, 0.7),
                    'volume_24h': np.random.randint(1000, 100000),
                    'open_interest': np.random.randint(5000, 50000),
                    'bid_ask_spread': np.random.uniform(0.01, 0.05),
                    'days_to_expiration': np.random.randint(1, 90),
                    'price_change_7d': price_change,
                    'volatility': np.random.uniform(0.01, 0.15),
                    'volume_trend': np.random.normal(0, 0.2)
                },
                'news_features': {
                    'article_count': np.random.randint(0, 30),
                    'avg_sentiment': sentiment,
                    'sentiment_std': np.random.uniform(0, 0.3),
                    'positive_ratio': max(0, min(1, 0.5 + sentiment)),
                    'negative_ratio': max(0, min(1, 0.5 - sentiment)),
                    'source_diversity': np.random.uniform(0.3, 1.0),
                    'recency_score': np.random.uniform(0, 1)
                }
            }
            
            data.append(data_point)
            labels.append(label)
        
        return data, labels
    
    # Generate and train
    print("Generating synthetic training data...")
    training_data, labels = generate_synthetic_data(200)
    
    print("Training classifier...")
    classifier = MarketClassifier(model_type='random_forest')
    metrics = classifier.train(training_data, labels)
    
    print(f"\nTraining Results:")
    print(f"Train Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"CV Mean: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
    
    print(f"\nTop 5 Important Features:")
    for i, (feature, importance) in enumerate(list(metrics['feature_importance'].items())[:5], 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Make a prediction
    print("\n\nMaking sample prediction...")
    sample_market = {
        'current_price': 0.65,
        'volume_24h': 50000,
        'price_change_7d': 0.15,
        'volatility': 0.05,
        'volume_trend': 0.1
    }
    
    sample_news = {
        'article_count': 15,
        'avg_sentiment': 0.4,
        'positive_ratio': 0.7,
        'negative_ratio': 0.1,
        'recency_score': 0.8
    }
    
    prediction = classifier.predict(sample_market, sample_news)
    print(f"\nPrediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Recommendation: {prediction['recommendation']}")
    
    print("\nClass Probabilities:")
    for cls, prob in prediction['probabilities'].items():
        print(f"  {cls}: {prob:.1%}")
