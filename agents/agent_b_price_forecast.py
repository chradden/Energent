import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import yfinance as yf

class TransformerPriceForecaster(nn.Module):
    """Transformer-based electricity price forecaster"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, output_size: int = 24):
        super(TransformerPriceForecaster, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        x = self.dropout(x)
        
        # Use last token for prediction
        x = x[:, -1, :]
        output = self.output_projection(x)
        
        return output

class LSTMPriceForecaster(nn.Module):
    """LSTM-based electricity price forecaster"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 24):
        super(LSTMPriceForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output

class AgentBPriceForecast:
    """Agent B: Electricity Price Forecasting"""
    
    def __init__(self, model_type: str = "transformer"):
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def get_entsoe_data(self, country_code: str = "DE") -> pd.DataFrame:
        """Fetch electricity price data from ENTSO-E Transparency Platform"""
        # Note: This requires registration and API key from ENTSO-E
        # For demo purposes, we'll use synthetic data
        
        try:
            # This would be the actual API call with proper authentication
            # url = f"https://transparency.entsoe.eu/api"
            # params = {
            #     "securityToken": "YOUR_API_KEY",
            #     "documentType": "A44",
            #     "in_Domain": f"10Y1001A1001A83F",  # Germany
            #     "out_Domain": f"10Y1001A1001A83F",
            #     "periodStart": "202401010000",
            #     "periodEnd": "202401020000"
            # }
            # response = requests.get(url, params=params)
            
            # For now, return synthetic data
            return self._generate_synthetic_price_data()
            
        except Exception as e:
            print(f"Error fetching ENTSO-E data: {e}")
            return self._generate_synthetic_price_data()
    
    def get_yahoo_finance_data(self, symbol: str = "^GSPC") -> pd.DataFrame:
        """Fetch market data from Yahoo Finance as proxy for energy prices"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="7d", interval="1h")
            
            # Convert to our format
            df = pd.DataFrame({
                'timestamp': data.index,
                'price': data['Close'].values,
                'volume': data['Volume'].values
            })
            
            return df
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            return self._generate_synthetic_price_data()
    
    def _generate_synthetic_price_data(self) -> pd.DataFrame:
        """Generate synthetic electricity price data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='H')
        
        # Base price with daily and weekly patterns
        base_price = 50 + 20 * np.sin(2 * np.pi * dates.hour / 24)  # Daily cycle
        base_price += 10 * np.sin(2 * np.pi * dates.dayofweek / 7)  # Weekly cycle
        
        # Add volatility and trends
        volatility = np.random.normal(0, 5, len(dates))
        trend = np.linspace(0, 10, len(dates))  # Slight upward trend
        
        price = base_price + volatility + trend
        
        # Add time features
        df = pd.DataFrame({
            'timestamp': dates,
            'price': price,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
            'is_peak_hour': ((dates.hour >= 8) & (dates.hour <= 20)).astype(int)
        })
        
        return df
    
    def prepare_data(self, price_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Combine features
        features = price_df[['price', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour']].values
        targets = price_df['price'].values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences (24 hours of history to predict next 24 hours)
        X, y = [], []
        for i in range(24, len(features_scaled) - 23):
            X.append(features_scaled[i-24:i])
            y.append(targets[i:i+24])
        
        return np.array(X), np.array(y)
    
    def train_transformer(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train Transformer model"""
        input_size = X.shape[2]
        self.model = TransformerPriceForecaster(input_size=input_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train LSTM model"""
        input_size = X.shape[2]
        self.model = LSTMPriceForecaster(input_size=input_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def train(self, price_df: pd.DataFrame):
        """Train the forecasting model"""
        print(f"Training {self.model_type.upper()} model...")
        
        X, y = self.prepare_data(price_df)
        
        if self.model_type == "transformer":
            self.train_transformer(X, y)
        elif self.model_type == "lstm":
            self.train_lstm(X, y)
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, price_df: pd.DataFrame) -> np.ndarray:
        """Predict electricity prices for next 24 hours"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_data(price_df)
        
        if len(X) > 0:
            X_tensor = torch.FloatTensor(X[-1:])  # Use last sequence
            
            with torch.no_grad():
                if self.model_type == "transformer":
                    prediction = self.model(X_tensor).numpy()[0]
                elif self.model_type == "lstm":
                    prediction = self.model(X_tensor).numpy()[0]
                else:
                    prediction = np.full(24, 50)  # Default prediction
                    
            return prediction
        else:
            return np.full(24, 50)  # Default prediction
    
    def evaluate(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        # Split data for evaluation
        split_idx = int(len(price_df) * 0.8)
        train_df = price_df.iloc[:split_idx]
        test_df = price_df.iloc[split_idx:]
        
        # Retrain on training data
        self.train(train_df)
        
        # Predict on test data
        predicted = self.predict(test_df)
        actual = test_df['price'].values[:24]  # First 24 hours
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }
    
    def get_current_prices(self) -> pd.DataFrame:
        """Get current electricity prices from multiple sources"""
        prices = {}
        
        # Try ENTSO-E first
        try:
            entsoe_data = self.get_entsoe_data()
            prices['entsoe'] = entsoe_data['price'].iloc[-24:].values
        except:
            prices['entsoe'] = None
        
        # Try Yahoo Finance as backup
        try:
            yahoo_data = self.get_yahoo_finance_data()
            prices['yahoo'] = yahoo_data['price'].iloc[-24:].values
        except:
            prices['yahoo'] = None
        
        # Use synthetic data as fallback
        if not any(prices.values()):
            synthetic_data = self._generate_synthetic_price_data()
            prices['synthetic'] = synthetic_data['price'].iloc[-24:].values
        
        return prices

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent_b = AgentBPriceForecast(model_type="transformer")
    
    # Get price data
    price_data = agent_b.get_entsoe_data()
    
    # Train model
    agent_b.train(price_data)
    
    # Make prediction
    forecast = agent_b.predict(price_data)
    print(f"24-hour electricity price forecast: {forecast}")
    
    # Save forecast to file
    forecast_df = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
        'electricity_price_forecast': forecast
    })
    forecast_df.to_csv('data/electricity_price_forecast.csv', index=False)
    
    # Evaluate model
    metrics = agent_b.evaluate(price_data)
    print(f"Model performance: {metrics}")
