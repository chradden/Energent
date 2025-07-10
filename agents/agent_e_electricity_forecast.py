import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional, Dict

class AgentEElectricityForecast:
    """Agent E: Electricity Consumption Forecasting (XGBoost, Weather, Holiday)"""
    def __init__(self, model_type: str = "xgboost", window_hours: int = 24, resolution_minutes: int = 60):
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.window_hours = window_hours
        self.resolution_minutes = resolution_minutes
        self.points_per_hour = 60 // self.resolution_minutes
        self.window_size = self.window_hours * self.points_per_hour

    @staticmethod
    def get_holidays_schleswig_holstein(year):
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/DE"
        try:
            response = requests.get(url)
            response.raise_for_status()
            holidays_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching holidays for {year}: {e}")
            return set()
        schleswig_holstein_holidays = set()
        for holiday in holidays_data:
            holiday_date_str = holiday['date']
            dt_object = datetime.strptime(holiday_date_str, '%Y-%m-%d').date()
            if holiday['global'] or (holiday.get('counties') and 'DE-SH' in holiday['counties']):
                schleswig_holstein_holidays.add(dt_object)
        return schleswig_holstein_holidays

    @staticmethod
    def get_weather_data(df, lat: float, lon: float, resolution_minutes: int = 60):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTimeIndex.")
        start = df.index.min()
        end = df.index.max()
        # Handle NaT and DatetimeIndex cases
        if pd.isnull(start) or pd.isnull(end):
            raise ValueError("DataFrame index has no valid timestamps.")
        try:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
        except Exception:
            raise ValueError("Could not convert start/end to Timestamp.")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "start_date": start_dt.strftime('%Y-%m-%d'),
            "end_date": end_dt.strftime('%Y-%m-%d'),
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
            weather_15min = weather_df.resample(f'{resolution_minutes}min').interpolate('linear')
            weather_aligned = weather_15min.reindex(df.index, method='nearest')
            weather_aligned['hour'] = [ts.hour for ts in weather_aligned.index]
            weather_aligned['day_of_week'] = [ts.weekday() for ts in weather_aligned.index]
            weather_aligned['month'] = [ts.month for ts in weather_aligned.index]
            weather_aligned['is_weekend'] = [1 if ts.weekday() in [5, 6] else 0 for ts in weather_aligned.index]
            return weather_aligned
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_day_time(hour):
        if 0 <= hour < 4:
            return 0  # Midnight
        elif 4 <= hour < 8:
            return 1  # Early Morning
        elif 8 <= hour < 12:
            return 2  # Morning
        elif 12 <= hour < 16:
            return 3  # Afternoon
        elif 16 <= hour < 20:
            return 4  # Evening
        else:
            return 5  # Night

    def load_electricity_from_csv(self, path: str) -> pd.DataFrame:
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            header_idx = next(i for i, line in enumerate(lines) if 'Einheit' in line)
        df = pd.read_csv(path, skiprows=header_idx+1, header=None, sep=';')
        df = df.iloc[:, :2]
        df.columns = ['datetime', 'kwh']
        df['kwh'] = df['kwh'].astype(str).str.replace(',', '.').astype(float)
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, format='%d.%m.%Y %H:%M')
        df = df.set_index('datetime')
        return df

    def prepare_features(self, df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Use list comprehensions for robust date extraction
        df['date'] = [d.date() for d in df.index]
        df['day'] = [d.weekday() + 1 for d in df.index]
        min_year = min(d.year for d in df.index)
        max_year = max(d.year for d in df.index)
        all_holidays = set()
        for year in range(min_year, max_year + 1):
            all_holidays.update(self.get_holidays_schleswig_holstein(year))
        df['Holiday'] = [1 if d.date() in all_holidays else 0 for d in df.index]
        weather_features_df = self.get_weather_data(df, lat=lat, lon=lon, resolution_minutes=self.resolution_minutes)
        if not weather_features_df.empty:
            df = df.merge(weather_features_df, left_index=True, right_index=True, how='left')
        df['day_time'] = [d.hour for d in df.index]
        df['day_time'] = df['day_time'].map(self.get_day_time)
        temp_bins = [-float('inf'), 0, 10, 20, float('inf')]
        temp_labels = ['Very Cold', 'Cold', 'Normal', 'Hot']
        df['temp_category'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False)
        temp_category_map = {'Very Cold': 0, 'Cold': 1, 'Normal': 2, 'Hot': 3}
        df['temp_category_int'] = df['temp_category'].map(lambda x: temp_category_map.get(x, np.nan))
        df['month'] = [d.month for d in df.index]
        df['is_weekend'] = [1 if d.weekday() in [5, 6] else 0 for d in df.index]
        return df

    def train_xgboost(self, df: pd.DataFrame):
        final_features = ['day', 'temp_category_int', 'day_time', 'month', 'kwh', 'Holiday', 'temperature', 'is_weekend']
        df_final = df[final_features].dropna().copy()
        X = df_final.drop('kwh', axis=1).values
        y = df_final['kwh'].values
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        final_features = ['day', 'temp_category_int', 'day_time', 'month', 'Holiday', 'temperature', 'is_weekend']
        df_pred = df[final_features].dropna().copy()
        X = df_pred.values
        X_scaled = self.scaler.transform(X)
        if self.model is None:
            raise ValueError("XGBoost model is not initialized. Please train the model before prediction.")
        return self.model.predict(X_scaled)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        y_true = df['kwh'].values
        y_pred = self.predict(df)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'mae': mae, 'rmse': rmse, 'mape': mape}

    def train_from_csv(self, csv_path: str, lat: float, lon: float):
        df = self.load_electricity_from_csv(csv_path)
        df = self.prepare_features(df, lat, lon)
        self.train_xgboost(df)

    def predict_from_csv(self, csv_path: str, lat: float, lon: float) -> np.ndarray:
        df = self.load_electricity_from_csv(csv_path)
        df = self.prepare_features(df, lat, lon)
        return self.predict(df)

    def get_future_weather_data(self, periods: int = 168, freq: str = '60min', lat: float = 54.3233, lon: float = 10.1228) -> pd.DataFrame:
        """Fetch weather forecast for the next 7 days at the given interval (default: hourly)."""
        now = datetime.now()
        target_index = pd.date_range(start=now, periods=periods, freq=freq)
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
            # Resample/interpolate to desired interval
            weather_resampled = weather_df.resample(freq).interpolate('linear')
            weather_resampled = weather_resampled.reindex(target_index, method='nearest')
            # Add time features
            weather_resampled['hour'] = weather_resampled.index.hour
            weather_resampled['day_of_week'] = weather_resampled.index.dayofweek
            weather_resampled['month'] = weather_resampled.index.month
            weather_resampled['is_weekend'] = weather_resampled['day_of_week'].isin([5, 6]).astype(int)
            if weather_resampled.empty:
                return pd.DataFrame(index=target_index)
            return weather_resampled
        except Exception as e:
            print(f"Error fetching future weather data: {e}")
            # Fallback: generate synthetic weather
            return pd.DataFrame(index=target_index)

    def predict_next_7_days(self, lat: float = 54.3233, lon: float = 10.1228) -> tuple:
        """Predict electricity consumption for the next 7 days (hourly intervals). Returns (forecast_df, features_df)."""
        periods = 7 * 24  # 7 days, hourly
        freq = '60min'
        # Get future weather
        weather_df = self.get_future_weather_data(periods=periods, freq=freq, lat=lat, lon=lon)
        if weather_df is None or weather_df.empty:
            raise ValueError("Could not fetch future weather data.")
        # Add holiday and other features
        df = weather_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df['date'] = [d.date() for d in df.index]
        # Use only valid Timestamp objects for min/max year
        years = [d.year for d in df.index if isinstance(d, pd.Timestamp) and not pd.isna(d) and d is not pd.NaT]
        if not years:
            raise ValueError("No valid years in index for holiday calculation.")
        min_year = min(years)
        max_year = max(years)
        df['day'] = [d.weekday() + 1 for d in df.index]
        all_holidays = set()
        for year in range(min_year, max_year + 1):
            all_holidays.update(self.get_holidays_schleswig_holstein(year))
        df['Holiday'] = [1 if d.date() in all_holidays else 0 for d in df.index]
        df['day_time'] = [d.hour for d in df.index]
        df['day_time'] = df['day_time'].map(self.get_day_time)
        temp_bins = [-float('inf'), 0, 10, 20, float('inf')]
        temp_labels = ['Very Cold', 'Cold', 'Normal', 'Hot']
        df['temp_category'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False)
        temp_category_map = {'Very Cold': 0, 'Cold': 1, 'Normal': 2, 'Hot': 3}
        df['temp_category_int'] = df['temp_category'].map(lambda x: temp_category_map.get(x, np.nan))
        df['month'] = [d.month for d in df.index]
        df['is_weekend'] = [1 if d.weekday() in [5, 6] else 0 for d in df.index]
        # Predict
        final_features = ['day', 'temp_category_int', 'day_time', 'month', 'Holiday', 'temperature', 'is_weekend']
        df_pred = df[final_features].dropna().copy()
        X = df_pred.values
        X_scaled = self.scaler.transform(X) if self.is_trained else X  # Avoid error if not trained
        if self.model is not None and self.is_trained:
            y_pred = self.model.predict(X_scaled)
        else:
            y_pred = np.full(len(df_pred), np.nan)
        forecast_df = pd.DataFrame({
            'timestamp': df_pred.index,
            'electricity_consumption_forecast': y_pred
        }).set_index('timestamp')
        # Return both forecast and the input features used
        return forecast_df, df_pred

# Example usage:
# agent = AgentEElectricityForecast()
# agent.train_from_csv('data/electricity consumption_2024-01-01.csv', lat=54.3233, lon=10.1228)
# predictions = agent.predict_from_csv('data/electricity consumption_2024-01-01.csv', lat=54.3233, lon=10.1228) 