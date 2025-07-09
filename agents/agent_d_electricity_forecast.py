import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def get_holidays_schleswig_holstein(year):
    """
    Fetches public holidays for Germany from Nager.Date API and filters for Schleswig-Holstein.
    Returns a set of datetime.date objects for holidays.
    """
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

def get_weather_data(heat_demand: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch weather data for the same time range and resolution as the heat_demand DataFrame.
    Assumes heat_demand has a DateTimeIndex.
    """
    if not isinstance(heat_demand.index, pd.DatetimeIndex):
        raise ValueError("heat_demand must have a DateTimeIndex.")

    start = heat_demand.index.min()
    end = heat_demand.index.max()

    if pd.isna(start) or pd.isna(end):
        raise ValueError("heat_demand index has no valid timestamps.")

    target_index = heat_demand.index

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "start_date": start.strftime('%Y-%m-%d'),
        "end_date": end.strftime('%Y-%m-%d'),
        "timezone": "Europe/Berlin"
    }

    print(f"\nFetching weather data for {lat}, {lon} from {params['start_date']} to {params['end_date']}...")

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

        weather_15min = weather_df.resample('15min').interpolate('linear')
        weather_aligned = weather_15min.reindex(target_index, method='nearest')

        weather_aligned['hour'] = weather_aligned.index.hour
        weather_aligned['day_of_week'] = weather_aligned.index.dayofweek
        weather_aligned['month'] = weather_aligned.index.month
        weather_aligned['is_weekend'] = weather_aligned['day_of_week'].isin([5, 6]).astype(int)

        print("Weather data fetched and processed successfully.")
        return weather_aligned

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during weather data processing: {e}")
        return pd.DataFrame()

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

def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

def evaluate_model(model, data_loader, scaler_obj, target_idx, original_scaled_data_for_inverse_transform, device):
    model.eval()
    predictions_scaled = []
    actual_scaled = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions_scaled.extend(outputs.cpu().numpy().flatten())
            actual_scaled.extend(y_batch.cpu().numpy().flatten())

    y_pred_original = inverse_transform_kwh(np.array(predictions_scaled).reshape(-1, 1),
                                            original_scaled_data_for_inverse_transform,
                                            target_idx, scaler_obj)

    y_actual_original = inverse_transform_kwh(np.array(actual_scaled).reshape(-1, 1),
                                              original_scaled_data_for_inverse_transform,
                                              target_idx, scaler_obj)

    mae = mean_absolute_error(y_actual_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_actual_original, y_pred_original))
    return mae, rmse, y_pred_original, y_actual_original

def inverse_transform_kwh(scaled_predictions, original_scaled_data, target_idx, scaler_obj):
    dummy_array = np.zeros((scaled_predictions.shape[0], original_scaled_data.shape[1]))
    dummy_array[:, target_idx] = scaled_predictions.flatten()
    original_values = scaler_obj.inverse_transform(dummy_array)
    return original_values[:, target_idx]

def prepare_data(csv_path, lat=54.3233, lon=10.1228):
    """
    Prepare data for electricity consumption forecasting.
    
    Args:
        csv_path (str): Path to the CSV file with electricity consumption data
        lat (float): Latitude for weather data
        lon (float): Longitude for weather data
    
    Returns:
        tuple: (train_data, val_data, test_data, scaler, target_idx, all_cols_for_lstm_scaling)
    """
    # Read the CSV File
    with open(csv_path, encoding='utf-8') as f:
        lines = f.readlines()
        header_idx = next(i for i, line in enumerate(lines) if 'Einheit' in line)
    
    df = pd.read_csv(csv_path, skiprows=header_idx+1, header=None, sep=';')
    df = df.iloc[:, :2]
    df.columns = ['datetime', 'kwh']
    df['kwh'] = df['kwh'].astype(str).str.replace(',', '.').astype(float)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, format='%d.%m.%Y %H:%M')
    df = df.set_index('datetime')

    # Adding additional features
    df['date'] = df.index.date
    df['day'] = df.index.dayofweek + 1

    # Get holidays
    min_year = df.index.min().year
    max_year = df.index.max().year
    all_holidays_schleswig_holstein = set()
    for year in range(min_year, max_year + 1):
        all_holidays_schleswig_holstein.update(get_holidays_schleswig_holstein(year))
    
    df['Holiday'] = pd.Series(df.index.date).map(lambda x: 1 if x in all_holidays_schleswig_holstein else 0).values

    # Get weather data
    weather_features_df = get_weather_data(df, lat=lat, lon=lon)
    if not weather_features_df.empty:
        df = df.merge(weather_features_df, left_index=True, right_index=True, how='left')

    df['day_time'] = df.index.hour.map(get_day_time)

    temp_bins = [-float('inf'), 0, 10, 20, float('inf')]
    temp_labels = ['Very Cold', 'Cold', 'Normal', 'Hot']
    df['temp_category'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False)
    temp_category_map = {'Very Cold': 0, 'Cold': 1, 'Normal': 2, 'Hot': 3}
    df['temp_category_int'] = df['temp_category'].map(temp_category_map)

    # Select final features
    final_features = ['day', 'temp_category_int', 'day_time', 'month', 'kwh', 'Holiday', 'temperature', 'is_weekend']
    df_final = df[final_features].copy()
    
    # Handle missing values
    df_final = df_final.fillna(method='ffill').fillna(method='bfill')
    
    # Split data
    train_size = int(len(df_final) * 0.7)
    val_size = int(len(df_final) * 0.15)
    
    train_df = df_final[:train_size]
    val_df = df_final[train_size:train_size + val_size]
    test_df = df_final[train_size + val_size:]
    
    # Prepare features and target
    TARGET = 'kwh'
    FEATURES = [col for col in final_features if col != TARGET]
    
    all_cols_for_lstm_scaling = FEATURES + [TARGET]
    
    # Scale the data
    scaler = StandardScaler()
    train_scaled_data = scaler.fit_transform(train_df[all_cols_for_lstm_scaling])
    val_scaled_data = scaler.transform(val_df[all_cols_for_lstm_scaling])
    test_scaled_data = scaler.transform(test_df[all_cols_for_lstm_scaling])
    
    target_idx = all_cols_for_lstm_scaling.index(TARGET)
    
    return train_scaled_data, val_scaled_data, test_scaled_data, scaler, target_idx, all_cols_for_lstm_scaling, test_df

def train_lstm_model(train_scaled_data, val_scaled_data, test_scaled_data, target_idx, all_cols_for_lstm_scaling, 
                    hidden_size=50, dropout_rate=0.2, num_epochs=50, batch_size=32, look_back=24):
    """
    Train LSTM model for electricity consumption forecasting.
    
    Args:
        train_scaled_data: Training data
        val_scaled_data: Validation data  
        test_scaled_data: Test data
        target_idx: Index of target column
        all_cols_for_lstm_scaling: List of all column names
        hidden_size: LSTM hidden size
        dropout_rate: Dropout rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        look_back: Look back window for LSTM
    
    Returns:
        tuple: (trained_model, scaler, target_idx, all_cols_for_lstm_scaling)
    """
    # Prepare sequences
    X_train_lstm_full = train_scaled_data[:, :-1]  # All features except target
    y_train_lstm_scaled = train_scaled_data[:, target_idx]  # Target
    
    X_val_lstm_full = val_scaled_data[:, :-1]
    y_val_lstm_scaled = val_scaled_data[:, target_idx]
    
    X_test_lstm_full = test_scaled_data[:, :-1]
    y_test_lstm_scaled = test_scaled_data[:, target_idx]
    
    # Create sequences
    X_train_lstm, y_train_lstm = create_sequences(X_train_lstm_full, y_train_lstm_scaled, look_back)
    X_val_lstm, y_val_lstm = create_sequences(X_val_lstm_full, y_val_lstm_scaled, look_back)
    X_test_lstm, y_test_lstm = create_sequences(X_test_lstm_full, y_test_lstm_scaled, look_back)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_lstm, dtype=torch.float32).unsqueeze(1)
    
    X_val_tensor = torch.tensor(X_val_lstm, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_lstm, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_lstm, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train_lstm.shape[2]
    output_size = 1
    
    model = LSTMModel(input_size, hidden_size, output_size, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                predictions_val = model(batch_X_val)
                loss_val = criterion(predictions_val, batch_y_val)
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return model, test_loader, val_scaled_data, test_scaled_data

def get_electricity_forecast(csv_path, lat=54.3233, lon=10.1228, forecast_hours=24):
    """
    Get electricity consumption forecast.
    
    Args:
        csv_path (str): Path to the CSV file with electricity consumption data
        lat (float): Latitude for weather data (default: Kiel)
        lon (float): Longitude for weather data (default: Kiel)
        forecast_hours (int): Number of hours to forecast
    
    Returns:
        pandas.DataFrame: DataFrame with datetime index and forecasted consumption values
    """
    try:
        # Prepare data
        train_scaled_data, val_scaled_data, test_scaled_data, scaler, target_idx, all_cols_for_lstm_scaling, test_df = prepare_data(csv_path, lat, lon)
        
        # Train model
        model, test_loader, val_scaled_data, test_scaled_data = train_lstm_model(
            train_scaled_data, val_scaled_data, test_scaled_data, target_idx, all_cols_for_lstm_scaling
        )
        
        # Get predictions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        predictions_scaled = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                predictions_scaled.extend(outputs.cpu().numpy().flatten())
        
        # Inverse transform predictions
        y_pred_original = inverse_transform_kwh(
            np.array(predictions_scaled).reshape(-1, 1),
            test_scaled_data,
            target_idx,
            scaler
        )
        
        # Create forecast DataFrame
        # Use the test data timestamps for the predictions
        forecast_df = pd.DataFrame({
            'datetime': test_df.index[24:],  # Skip first 24 hours due to look_back
            'forecasted_consumption_kwh': y_pred_original
        }).set_index('datetime')
        
        return forecast_df
        
    except Exception as e:
        print(f"Error in electricity forecast: {e}")
        return pd.DataFrame()

# Example usage (only runs if script is executed directly)
if __name__ == "__main__":
    # Example path - adjust as needed
    csv_path = r'C:\Users\Christian\Coding\Energent\Energent\data\electricity consumption_2024-01-01.csv'
    
    if os.path.exists(csv_path):
        forecast_df = get_electricity_forecast(csv_path)
        print("Forecast completed successfully!")
        print(forecast_df.head())
    else:
        print(f"CSV file not found at: {csv_path}")