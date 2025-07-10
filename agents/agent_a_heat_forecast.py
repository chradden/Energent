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
import os
import joblib

class LSTMHeatForecaster(nn.Module):
    """LSTM-based heat demand forecaster"""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 96):
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
    def __init__(self, model_type: str = "lstm", device: Optional[str] = None, batch_size: int = 64, window_hours: int = 24, resolution_minutes: int = 15):
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.batch_size = batch_size
        self.window_hours = window_hours
        self.resolution_minutes = resolution_minutes
        self.points_per_hour = 60 // self.resolution_minutes
        self.window_size = self.window_hours * self.points_per_hour
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
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
            'humidity': 55 + np.random.normal(0, 10, len(dates)),
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
    
    def prepare_data(
        self,
        weather_df: pd.DataFrame,
        heat_demand: Optional[pd.Series] = None,
        window_hours: Optional[int] = None,
        resolution_minutes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training, also returns the timestamps for each target window."""
        if heat_demand is None:
            heat_demand = self._generate_synthetic_heat_demand(weather_df)
        features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        targets = heat_demand.values
        timestamps = weather_df.index
        wh = window_hours if window_hours is not None else self.window_hours
        rm = resolution_minutes if resolution_minutes is not None else self.resolution_minutes
        points_per_hour = 60 // rm
        window_size = wh * points_per_hour
        # Only scale features
        features_scaled = self.scaler.fit_transform(features)
        X, y, y_timestamps = [], [], []
        for i in range(window_size, len(features_scaled) - window_size + 1):
            X.append(features_scaled[i-window_size:i])
            y.append(targets[i:i+window_size])
            y_timestamps.append(timestamps[i:i+window_size])
        return np.array(X), np.array(y), np.array(y_timestamps)

    def prepare_data_weather_only(self, weather_df: pd.DataFrame, heat_demand: Optional[pd.Series] = None, window_hours: int = 24, resolution_minutes: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data using only weather features to predict heat demand."""
        if heat_demand is None:
            heat_demand = self._generate_synthetic_heat_demand(weather_df)
        features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        targets = heat_demand.values
        timestamps = weather_df.index
        points_per_hour = 60 // resolution_minutes
        window_size = window_hours * points_per_hour
        features_scaled = self.scaler.fit_transform(features)
        X, y, y_timestamps = [], [], []
        for i in range(window_size, len(features_scaled)):
            X.append(features_scaled[i-window_size:i])
            y.append(targets[i])
            y_timestamps.append(timestamps[i])
        return np.array(X), np.array(y), np.array(y_timestamps)

    def prepare_data_history_only(self, heat_demand: pd.Series, window_hours: int = 24, resolution_minutes: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data using only past heat demand to predict next value."""
        targets = np.asarray(heat_demand.values)
        timestamps = heat_demand.index
        points_per_hour = 60 // resolution_minutes
        window_size = window_hours * points_per_hour
        X, y, y_timestamps = [], [], []
        for i in range(window_size, len(targets)):
            X.append(targets[i-window_size:i])
            y.append(targets[i])
            y_timestamps.append(timestamps[i])
        X = np.array(X).reshape(-1, window_size, 1)  # shape: (samples, window, 1)
        return X, np.array(y), np.array(y_timestamps)

    def prepare_data_history_weather(self, weather_df: pd.DataFrame, heat_demand: pd.Series, window_hours: int = 24, resolution_minutes: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data using both past heat demand and current weather to predict next value."""
        features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        targets = np.asarray(heat_demand.values)
        timestamps = weather_df.index
        points_per_hour = 60 // resolution_minutes
        window_size = window_hours * points_per_hour
        features_scaled = self.scaler.fit_transform(features)
        X, y, y_timestamps = [], [], []
        for i in range(window_size, len(features_scaled)):
            # Concatenate past heat demand and current weather
            past_heat = targets[i-window_size:i].reshape(-1, 1)
            current_weather = features_scaled[i].reshape(1, -1)
            # Repeat current weather for each step in window (or just use at t?)
            # Here, concatenate past_heat and current_weather for each step
            combined = np.concatenate([past_heat, np.repeat(current_weather, window_size, axis=0)], axis=1)
            X.append(combined)
            y.append(targets[i])
            y_timestamps.append(timestamps[i])
        return np.array(X), np.array(y), np.array(y_timestamps)

    def prepare_weather_only_onetoone(self, weather_df: pd.DataFrame, target_series: pd.Series):
        """Prepare data: X = weather at t, y = heat at t (one-to-one)."""
        features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        targets = np.asarray(target_series.values)
        timestamps = weather_df.index
        min_len = min(len(features), len(targets))
        X = features[:min_len]
        y = targets[:min_len]
        y_times = timestamps[:min_len]
        return X, y, y_times

    def prepare_history_only_window_one(self, series: pd.Series, window_size: int = 96):
        """Prepare data: X = previous window_size heat values, y = heat at t (history only, window + one)."""
        heat_values = np.asarray(series.values)
        timestamps = series.index
        X, y, y_times = [], [], []
        for i in range(window_size, len(heat_values)):
            past_heat = heat_values[i-window_size:i]
            X.append(past_heat)
            y.append(heat_values[i])
            y_times.append(timestamps[i])
        X = np.array(X)  # shape: (samples, window_size)
        return X, np.array(y), np.array(y_times)

    def prepare_history_weather_window_one(self, series: pd.Series, weather_df: pd.DataFrame, window_size: int = 96):
        """Prepare data: X = previous window_size heat values + weather at t, y = heat at t (history + weather, window + one)."""
        heat_values = np.asarray(series.values)
        weather_features = weather_df[['temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 'is_weekend']].values
        timestamps = series.index
        X, y, y_times = [], [], []
        for i in range(window_size, len(heat_values)):
            past_heat = heat_values[i-window_size:i]
            weather_now = weather_features[i]
            X.append(np.concatenate([past_heat, weather_now]))
            y.append(heat_values[i])
            y_times.append(timestamps[i])
        X = np.array(X)  # shape: (samples, window_size + n_features)
        return X, np.array(y), np.array(y_times)

    def train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train LSTM model for single-step prediction (output_size=1)."""
        from torch.utils.data import DataLoader, TensorDataset
        input_size = X.shape[2]
        output_size = 1  # Single-step prediction
        self.model = LSTMHeatForecaster(input_size=input_size, output_size=output_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X)
        # Ensure y is shape (samples, 1)
        if len(y.shape) == 1:
            y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        else:
            y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataset):.4f}')
    
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
    
    def train_lstm_generic(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, input_size: Optional[int] = None, batch_size: Optional[int] = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.cuda.empty_cache()
        # Ensure X is 3D for LSTM
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        if input_size is None:
            input_size = X.shape[2]
        if batch_size is None:
            batch_size = 16  # Lower default batch size for memory efficiency
        self.model = LSTMHeatForecaster(input_size=input_size, output_size=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # shape: (samples, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataset):.4f}')

    def predict_lstm_generic(self, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise ValueError("LSTM model is not initialized. Please train the model before prediction.")
        self.model.eval()
        preds = []
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        # Ensure X is 3D for LSTM
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                pred = self.model(xb).cpu().numpy().flatten()
                preds.append(pred)
        return np.concatenate(preds)

    def train(self, weather_df: pd.DataFrame, heat_demand: Optional[pd.Series] = None):
        """Train the forecasting model"""
        print(f"Training {self.model_type.upper()} model...")
        if self.model_type == "lstm":
            X, y, _ = self.prepare_data(weather_df, heat_demand)
            self.train_lstm(X, y)
        elif self.model_type == "xgboost":
            X, y, _ = self.prepare_data(weather_df, heat_demand)
            self.train_xgboost(X, y)
        elif self.model_type == "prophet":
            if heat_demand is None:
                heat_demand = self._generate_synthetic_heat_demand(weather_df)
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
            X, y, _ = self.prepare_data(weather_df)
            if len(X) > 0:
                if self.model is None:
                    raise ValueError("LSTM model is not initialized. Please train the model before prediction.")
                X_tensor = torch.FloatTensor(X[-1:]).to(self.device)  # Use last sequence
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(X_tensor).cpu().numpy()[0]
                return prediction
            else:
                return np.full(self.window_size, 250)  # Default prediction
        elif self.model_type == "xgboost":
            X, y, _ = self.prepare_data(weather_df)
            if len(X) > 0:
                if self.model is None:
                    raise ValueError("XGBoost model is not initialized. Please train the model before prediction.")
                X_flat = X[-1:].reshape(1, -1)
                prediction = []
                for i in range(self.window_size):
                    pred = self.model.predict(X_flat)[0]
                    prediction.append(pred)
                    X_flat[0, i*6:(i+1)*6] = X_flat[0, (i-1)*6:i*6]  # Shift features
                prediction = np.array(prediction)
                return prediction
            else:
                return np.full(self.window_size, 250)
        elif self.model_type == "prophet":
            if self.model is None:
                raise ValueError("Prophet model is not initialized. Please train the model before prediction.")
            freq = f'{self.resolution_minutes}min'
            future = self.model.make_future_dataframe(periods=self.window_size, freq=freq)
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(self.window_size).values
        return np.full(self.window_size, 250)  # Fallback

    def evaluate(self, weather_df: pd.DataFrame, actual_demand: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predicted = self.predict(weather_df)
        actual = actual_demand.values[:self.window_size]  # First 24 hours
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }

    def evaluate_multiple_models(self, y_true: np.ndarray, preds_dict: dict) -> dict:
        """Evaluate multiple models' predictions. preds_dict: {name: y_pred}"""
        results = {}
        for name, y_pred in preds_dict.items():
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            results[name] = {'mae': mae, 'rmse': rmse, 'mape': mape}
        return results

    def load_heat_demand_from_csv(self, path: str) -> pd.Series:
        """Load heat demand from gas usage CSV (German format, semicolon, decimal comma)."""
        import pandas as pd
        # Find the header row with 'Einheit'
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            header_idx = next(i for i, line in enumerate(lines) if 'Einheit' in line)
        # Read the CSV, using semicolon as separator, no header
        df = pd.read_csv(path, skiprows=header_idx+1, header=None, sep=';')
        # Keep only the first two columns
        df = df.iloc[:, :2]
        df.columns = ['datetime', 'kwh']
        # Convert German decimal comma to dot and cast to float
        df['kwh'] = df['kwh'].astype(str).str.replace(',', '.').astype(float)
        # Convert datetime column to pandas datetime (dayfirst, and exact format)
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, format='%d.%m.%Y %H:%M')
        # Set datetime as index
        df = df.set_index('datetime')
        # Resample to 15-min intervals if needed (mean)
        df = df.resample(f'{self.resolution_minutes}min').mean().interpolate('linear')
        return df['kwh']

    def get_future_weather_data(self, periods: int = 672, freq: str = '15min', lat: float = 53.5511, lon: float = 9.9937) -> pd.DataFrame:
        """Fetch weather forecast for the next 7 days at 15-min intervals (default)."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        now = datetime.now()
        end = now + timedelta(minutes=periods * self.resolution_minutes)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "forecast_days": 7,
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
            # Resample/interpolate to 15-min intervals
            weather_15min = weather_df.resample(freq).interpolate('linear')
            # Limit to the next 7 days (periods)
            weather_15min = weather_15min.iloc[:periods]
            # Add time features
            weather_15min['hour'] = weather_15min.index.hour
            weather_15min['day_of_week'] = weather_15min.index.dayofweek
            weather_15min['month'] = weather_15min.index.month
            weather_15min['is_weekend'] = weather_15min['day_of_week'].isin([5, 6]).astype(int)
            return weather_15min
        except Exception as e:
            print(f"Error fetching future weather data: {e}")
            # Fallback: generate synthetic weather
            target_index = pd.date_range(start=now, periods=periods, freq=freq)
            return self._generate_synthetic_weather(target_index=target_index)

    def save_model(self, path: str):
        """Save model and scaler to file."""
        if self.model_type == "lstm":
            import torch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler
            }, path)
        elif self.model_type == "xgboost":
            joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        # Prophet not implemented

    def load_model(self, path: str):
        """Load model and scaler from file."""
        if self.model_type == "lstm":
            import torch
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
            self.model = LSTMHeatForecaster(input_size=input_size, output_size=1).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.is_trained = True
        elif self.model_type == "xgboost":
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
        # Prophet not implemented

    def train_from_csv(self, csv_path: str, lat: float = 53.5511, lon: float = 9.9937):
        """Train the model using heat demand from CSV and aligned weather data. Save model after training."""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"agent_a_{self.model_type}.pkl")
        heat_demand = self.load_heat_demand_from_csv(csv_path)
        weather_df = self.get_weather_data(heat_demand.to_frame(), lat=lat, lon=lon)
        X, y, _ = self.prepare_data_weather_only(weather_df, heat_demand, window_hours=self.window_hours, resolution_minutes=self.resolution_minutes)
        self.train_lstm(X, y)  # Now always single-step
        self.is_trained = True
        self.save_model(model_path)

    def try_load_or_train(self, csv_path: str, lat: float = 53.5511, lon: float = 9.9937):
        """Try to load model; if not found, train and save."""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"agent_a_{self.model_type}.pkl")
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_from_csv(csv_path, lat=lat, lon=lon)

    def predict_next_7_days(self, lat: float = 53.5511, lon: float = 9.9937) -> pd.DataFrame:
        """Predict heat demand for the next 7 days (672 points, 15-min intervals)."""
        periods = 7 * 24 * (60 // self.resolution_minutes)
        weather_df = self.get_future_weather_data(periods=periods, freq=f'{self.resolution_minutes}min', lat=lat, lon=lon)
        # Use the last available window for prediction
        X, _, timestamps = self.prepare_data_weather_only(weather_df, None, window_hours=self.window_hours, resolution_minutes=self.resolution_minutes)
        preds = []
        for i in range(len(X)):
            X_input = X[i:i+1]
            X_tensor = torch.FloatTensor(X_input).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy().flatten()[0]
            preds.append(pred)
        # preds now has one value per window, align with timestamps
        pred_df = pd.DataFrame({
            'timestamp': timestamps,
            'heat_demand_forecast': preds
        })
        return pred_df

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
        'timestamp': pd.date_range(start=datetime.now(), periods=agent_a.window_size, freq='H'),
        'heat_demand_forecast': forecast
    })
    forecast_df.to_csv('DATA/heat_demand_forecast.csv', index=False)


