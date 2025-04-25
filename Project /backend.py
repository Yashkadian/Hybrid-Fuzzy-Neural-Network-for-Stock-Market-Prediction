import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="FuzzyStock AI API",
    description="Hybrid Fuzzy Neural Network for Stock Prediction",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


nn_model = create_model()
scaler = MinMaxScaler()


class StockRequest(BaseModel):
    ticker: str


def get_stock_data(ticker):
   
    try:
        if not ticker.endswith('.NS'):
            ticker += '.NS'
        data = yf.download(ticker, period='3mo', progress=False)
        return data if len(data) >= 20 else None
    except Exception as e:
        logger.error(f"Failed to download {ticker}: {str(e)}")
        return None

def calculate_indicators(data):
   
    try:
        data = data.copy()
        
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        
        data['Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        return data.dropna()
    except Exception as e:
        logger.error(f"Indicator calculation failed: {str(e)}")
        return None

def neural_network_predict(data):
    
    try:
        features = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'Volatility']].tail(30)
        scaled = scaler.fit_transform(features)
        X = np.array([scaled.mean(axis=0)])
        return float(nn_model.predict(X, verbose=0)[0][0])
    except Exception as e:
        logger.error(f"Neural network prediction failed: {str(e)}")
        return 0.5

def generate_signal(data):
    
    try:
        latest = data.iloc[-1]
        
        
        close = latest['Close'].iloc[0] if hasattr(latest['Close'], 'iloc') else float(latest['Close'])
        sma20 = latest['SMA_20'].iloc[0] if hasattr(latest['SMA_20'], 'iloc') else float(latest['SMA_20'])
        sma50 = latest['SMA_50'].iloc[0] if hasattr(latest['SMA_50'], 'iloc') else float(latest['SMA_50'])
        rsi = latest['RSI'].iloc[0] if hasattr(latest['RSI'], 'iloc') else float(latest['RSI'])
        vol = latest['Volatility'].iloc[0] if hasattr(latest['Volatility'], 'iloc') else float(latest['Volatility'])
        
       
        nn_confidence = neural_network_predict(data)
        
        
        buy_conditions = [
            close > sma20,
            close > sma50,
            rsi < 70,
            vol < 30
        ]
        
        sell_conditions = [
            close < sma20,
            rsi > 70,
            vol > 40
        ]
        
        
        fuzzy_score = sum(buy_conditions) - sum(sell_conditions)
        combined_score = (fuzzy_score * 0.6) + (nn_confidence * 0.4)
        
        if combined_score > 0.7:
            return "BUY", close, f"{combined_score*100:.1f}%"
        elif combined_score < 0.3:
            return "SELL", close, f"{(1-combined_score)*100:.1f}%"
        return "HOLD", close, "50.0%"
        
    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        return "ERROR", None, "0%"

@app.post("/predict")
async def predict_stock(request: StockRequest):
    """Get stock prediction"""
    try:
        ticker = request.ticker
        data = get_stock_data(ticker)
        if data is None:
            return {"status": "error", "message": "Failed to get stock data"}
        
        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None:
            return {"status": "error", "message": "Indicator calculation failed"}
        
        signal, price, confidence = generate_signal(data_with_indicators)
        if signal == "ERROR":
            return {"status": "error", "message": "Signal generation failed"}
        
        latest = data_with_indicators.iloc[-1]
        
        
        def safe_format(value, prefix='₹'):
            try:
                val = value.iloc[0] if hasattr(value, 'iloc') else float(value)
                return f"{prefix}{val:.2f}" if prefix else f"{val:.1f}"
            except:
                return "N/A"
        
        return {
            "status": "success",
            "prediction": signal,
            "confidence": confidence,
            "price": safe_format(latest['Close']),
            "rsi": safe_format(latest['RSI'], ''),
            "volatility": safe_format(latest['Volatility'], '') + '%',
            "20D_MA": safe_format(latest['SMA_20']),
            "50D_MA": safe_format(latest['SMA_50'])
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"status": "error", "message": "Internal server error"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
