import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional

class LSTMHeatForecaster(nn.Module):
    """LSTM-based heat demand forecaster"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 24):
        super(LSTMHeatForecaster, self).__init__()
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

class AgentAHeatForecast:
    """Agent A: Heat Demand Forecasting"""
    
    def __init__(self, model_type: str = "lstm"):
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    # def get_weather_data(self, lat: float = 53.5511, lon: float = 9.9937) -> pd.DataFrame:
    #     """Fetch weather data from Open-Meteo API"""
    #     url = f"https://api.open-meteo.com/v1/forecast"
    #     params = {
    #         "latitude": lat,
    #         "longitude": lon,
    #         "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
    #         "timezone": "Europe/Berlin"
    #     }
        
    #     try:
    #         response = requests.get(url, params=params)
    #         response.raise_for_status()
    #         data = response.json()
            
    #         df = pd.DataFrame({
    #             'timestamp': pd.to_datetime(data['hourly']['time']),
    #             'temperature': data['hourly']['temperature_2m'],
    #             'humidity': data['hourly']['relative_humidity_2m'],
    #             'wind_speed': data['hourly']['wind_speed_10m'],
    #             'precipitation': data['hourly']['precipitation']
    #         })
            
    #         # Add time features
    #         df['hour'] = df['timestamp'].dt.hour
    #         df['day_of_week'] = df['timestamp'].dt.dayofweek
    #         df['month'] = df['timestamp'].dt.month
    #         df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
    #         return df
            
    #     except Exception as e:
    #         print(f"Error fetching weather data: {e}")
    #         return self._generate_synthetic_weather()
    
    def _generate_synthetic_weather(self, target_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Generate synthetic weather data for testing, aligned to the target_index if provided."""
        if target_index is not None:
            dates = target_index
        else:
            dates = pd.date_range(start=datetime.now(), periods=168, freq='H')

        # Synthetic temperature profile (daily cycle + seasonal trend)
        base_temp = 15 + 10 * np.sin(2 * np.pi * dates.hour / 24) + 5 * np.sin(2 * np.pi * dates.dayofyear / 365)
        temperature = base_temp + np.random.normal(0, 2, len(dates))

        df = pd.DataFrame({
            'timestamp': dates,
            'temperature': temperature,
            'humidity': 60 + np.random.normal(0, 10, len(dates)),
            'wind_speed': 5 + np.random.exponential(3, len(dates)),
            'precipitation': np.random.exponential(0.1, len(dates)),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': pd.Series(dates).dt.dayofweek.isin([5, 6]).astype(int).values
        })
        df = df.set_index('timestamp')
        return df
    
    def _generate_synthetic_heat_demand(self, weather_df: pd.DataFrame) -> pd.Series:
        """Generate synthetic heat demand based on weather and time patterns"""
        # Base demand varies by hour and day type
        base_demand = 200 + 100 * np.sin(2 * np.pi * weather_df['hour'] / 24)
        
        # Weekend effect
        weekend_factor = 0.8 if weather_df['is_weekend'].iloc[0] else 1.0
        
        # Temperature effect (higher demand when colder)
        temp_factor = 1.5 - 0.02 * weather_df['temperature']
        
        # Random variation
        noise = np.random.normal(0, 20, len(weather_df))
        

        demand = (base_demand * weekend_factor * temp_factor + noise).clip(50, 500)
        return pd.Series(demand, index=weather_df.index)

    def get_weather_data(self, heat_demand: pd.DataFrame, lat: float = 53.5511, lon: float = 9.9937) -> pd.DataFrame:
        """
        Fetch weather data for the same time range and resolution as the heat_demand DataFrame.
        Assumes heat_demand has a DateTimeIndex or a 'timestamp' column.
        """
        # Determine time range
        if isinstance(heat_demand.index, pd.DatetimeIndex):
            start = heat_demand.index.min()
            end = heat_demand.index.max()
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                if pd.isna(start) or pd.isna(end):
                    raise ValueError("heat_demand index has no valid timestamps.")
            else:
                raise ValueError("heat_demand index min/max did not return a Timestamp.")
            target_index = heat_demand.index
        elif 'timestamp' in heat_demand.columns:
            timestamps = pd.to_datetime(heat_demand['timestamp'])
            start = timestamps.min()
            end = timestamps.max()
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                if pd.isna(start) or pd.isna(end):
                    raise ValueError("heat_demand['timestamp'] has no valid timestamps.")
            else:
                raise ValueError("heat_demand['timestamp'] min/max did not return a Timestamp.")
            target_index = timestamps
        else:
            raise ValueError("heat_demand must have a DateTimeIndex or a 'timestamp' column.")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "start_date": start.strftime('%Y-%m-%d'),
            "end_date": end.strftime('%Y-%m-%d'),
            "timezone": "Europe/Berlin"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            weather_df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'wind_speed': data['hourly']['wind_speed_10m'],
                'precipitation': data['hourly']['precipitation']
            }).set_index('timestamp')
            

            # Resample/interpolate to 15-min intervals using '15min' instead of '15T'
            weather_15min = weather_df.resample('15min').interpolate('linear')

            # Align to heat_demand's timestamps
            weather_aligned = weather_15min.reindex(target_index, method='nearest')

            # Add time features if needed
            weather_aligned['hour'] = weather_aligned.index.hour
            weather_aligned['day_of_week'] = weather_aligned.index.dayofweek
            weather_aligned['month'] = weather_aligned.index.month
            weather_aligned['is_weekend'] = weather_aligned['day_of_week'].isin([5, 6]).astype(int)

            return weather_aligned

        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Ensure target_index is a DatetimeIndex
            if not isinstance(target_index, pd.DatetimeIndex):
                target_index = pd.DatetimeIndex(target_index)
            return self._generate_synthetic_weather(target_index=target_index)
    
    def prepare_data(self, weather_df: pd.DataFrame, heat_demand: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        if heat_demand is None:
            heat_demand = self._generate_synthetic_heat_demand(weather_df)
        
        # Combine features
        features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        targets = heat_demand.values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences (24 hours of history to predict next 24 hours)
        X, y = [], []
        for i in range(24, len(features_scaled) - 23):
            X.append(features_scaled[i-24:i])
            y.append(targets[i:i+24])
        
        return np.array(X), np.array(y)


    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train LSTM model"""
        input_size = X.shape[2]
        self.model = LSTMHeatForecaster(input_size=input_size)
        
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
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model"""
        # Reshape for XGBoost (flatten time dimension)
        X_flat = X.reshape(X.shape[0], -1)
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_flat, y[:, 0])  # Predict first hour, then iterate
    
    def train_prophet(self, timestamps: pd.DatetimeIndex, heat_demand: pd.Series):
        """Train Prophet model"""
        df_prophet = pd.DataFrame({
            'ds': timestamps,
            'y': heat_demand.values
        })
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        self.model.fit(df_prophet)
    
    def train(self, weather_df: pd.DataFrame, heat_demand: Optional[pd.Series] = None):
        """Train the forecasting model"""
        print(f"Training {self.model_type.upper()} model...")
        if self.model_type == "lstm":
            X, y = self.prepare_data(weather_df, heat_demand)
            self.train_lstm(X, y)
        elif self.model_type == "xgboost":
            X, y = self.prepare_data(weather_df, heat_demand)
            self.train_xgboost(X, y)
        elif self.model_type == "prophet":
            if heat_demand is None:
                heat_demand = self._generate_synthetic_heat_demand(weather_df)
            # Ensure timestamps is a DatetimeIndex
            if isinstance(weather_df.index, pd.DatetimeIndex):
                timestamps = weather_df.index
            else:
                timestamps = pd.DatetimeIndex(pd.to_datetime(weather_df['timestamp']))
            self.train_prophet(timestamps, heat_demand)
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Predict heat demand for next 24 hours"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.model_type == "lstm":
            X, _ = self.prepare_data(weather_df)
            if len(X) > 0:
                X_tensor = torch.FloatTensor(X[-1:])  # Use last sequence
                with torch.no_grad():
                    prediction = self.model(X_tensor).numpy()[0]
                return prediction
            else:
                return np.full(24, 250)  # Default prediction
                
        elif self.model_type == "xgboost":
            X, _ = self.prepare_data(weather_df)
            if len(X) > 0:
                X_flat = X[-1:].reshape(1, -1)
                prediction = []
                for i in range(24):
                    pred = self.model.predict(X_flat)[0]
                    prediction.append(pred)
                    # Update features for next prediction (simplified)
                    X_flat[0, i*6:(i+1)*6] = X_flat[0, (i-1)*6:i*6]  # Shift features
                return np.array(prediction)
            else:
                return np.full(24, 250)
                
        elif self.model_type == "prophet":
            future = self.model.make_future_dataframe(periods=24, freq='H')
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(24).values
        
        return np.full(24, 250)  # Fallback
    
    def evaluate(self, weather_df: pd.DataFrame, actual_demand: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predicted = self.predict(weather_df)
        actual = actual_demand.values[:24]  # First 24 hours
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent_a = AgentAHeatForecast(model_type="lstm")
    
    # Get weather data
    weather_data = agent_a.get_weather_data()
    
    # Train model
    agent_a.train(weather_data)
    
    # Make prediction
    forecast = agent_a.predict(weather_data)
    print(f"24-hour heat demand forecast: {forecast}")
    
    # Save forecast to file
    forecast_df = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
        'heat_demand_forecast': forecast
    })
    forecast_df.to_csv('DATA/heat_demand_forecast.csv', index=False)


