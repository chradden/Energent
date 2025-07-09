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
from sklearn.preprocessing import StandardScaler, MinMaxScaler # No OneHotEncoder needed
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression # Changed from RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


#### Read the CSV File
path = '/Users/saimohitvanapala/Documents/OpenCampus/ENERGENT/Heat forecast/electricity consumption_2024-01-01.csv'
with open(path, encoding='utf-8') as f:
    lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if 'Einheit' in line)
df = pd.read_csv(path, skiprows=header_idx+1, header=None, sep=';')
df = df.iloc[:, :2]
df.columns = ['datetime', 'kwh']
df['kwh'] = df['kwh'].astype(str).str.replace(',', '.').astype(float)
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, format='%d.%m.%Y %H:%M')
df = df.set_index('datetime')

# Adding additional features
df['date'] = df.index.date
df['day'] = df.index.dayofweek + 1


### Adding public holidays for Schleswig-Holstein
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

# Get the range of years present in your data's index
min_year = df.index.min().year
max_year = df.index.max().year

print(f"\nFetching holidays for years from {min_year} to {max_year}...")
all_holidays_schleswig_holstein = set()
for year in range(min_year, max_year + 1):
    all_holidays_schleswig_holstein.update(get_holidays_schleswig_holstein(year))

print(f"Total unique holidays fetched: {len(all_holidays_schleswig_holstein)}")
df['Holiday'] = pd.Series(df.index.date).map(lambda x: 1 if x in all_holidays_schleswig_holstein else 0).values

# Adding weather features

def get_weather_data(heat_demand: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch weather data for the same time range and resolution as the heat_demand DataFrame.
    Assumes heat_demand has a DateTimeIndex.
    """
    # Determine time range from the DataFrame's index (which is already datetime)
    if not isinstance(heat_demand.index, pd.DatetimeIndex):
        raise ValueError("heat_demand must have a DateTimeIndex.")

    start = heat_demand.index.min()
    end = heat_demand.index.max()

    # Ensure start and end are valid Timestamps
    if pd.isna(start) or pd.isna(end):
        raise ValueError("heat_demand index has no valid timestamps.")

    target_index = heat_demand.index

    # Open-Meteo Archive API URL
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
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Create DataFrame from fetched hourly weather data
        weather_df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'wind_speed': data['hourly']['wind_speed_10m'],
            'precipitation': data['hourly']['precipitation']
        }).set_index('timestamp')

        weather_15min = weather_df.resample('15min').interpolate('linear')
        weather_aligned = weather_15min.reindex(target_index, method='nearest')

        # Add time features
        weather_aligned['hour'] = weather_aligned.index.hour
        weather_aligned['day_of_week'] = weather_aligned.index.dayofweek
        weather_aligned['month'] = weather_aligned.index.month
        weather_aligned['is_weekend'] = weather_aligned['day_of_week'].isin([5, 6]).astype(int)

        print("Weather data fetched and processed successfully.")
        return weather_aligned

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error
    except Exception as e:
        print(f"An unexpected error occurred during weather data processing: {e}")
        return pd.DataFrame()

# Define coordinates for Kiel, Schleswig-Holstein
kiel_lat = 54.3233
kiel_lon = 10.1228

# Get weather data using your function
weather_features_df = get_weather_data(df, lat=kiel_lat, lon=kiel_lon)

# Check if weather_features_df is not empty before merging
if not weather_features_df.empty:
    df = df.merge(weather_features_df, left_index=True, right_index=True, how='left')

    print("\nDataFrame after adding weather features:")
    #print(df.head())
else:
    print("\nWeather data could not be fetched or processed. DataFrame not merged.")

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

df['day_time'] = df.index.hour.map(get_day_time)

temp_bins = [-float('inf'), 0, 10, 20, float('inf')]
temp_labels = ['Very Cold', 'Cold', 'Normal', 'Hot']

# Create a new column for temperature category
df['temp_category'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False)
temp_category_map = {
    'Very Cold': 0,
    'Cold': 1,
    'Normal': 2,
    'Hot': 3
}
df['temp_category_int'] = df['temp_category'].map(temp_category_map)
final_csv= df[['day', 'temp_category_int', 'day_time', 'month', 'kwh', 'Holiday', 'temperature', 'is_weekend']]
final_csv.to_csv('/Users/saimohitvanapala/Documents/OpenCampus/ENERGENT/Electricity_compustion/final.csv', index=False)

# Visualizing the electricity consumption data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['kwh'], label='Electricity Consumption (kWh)', color='blue')
plt.title('Electricity Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Visualizing the average electricity consumption on holidays vs non-holidays
plt.figure(figsize=(8, 6))
sns.barplot(x='Holiday', y='kwh', data=df, errorbar='sd', palette='viridis')
plt.title('Average Electricity Consumption: Holidays vs. Non-Holidays')
plt.xlabel('Holiday (0: No, 1: Yes)')
plt.ylabel('Average kWh Consumption')
plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualizing the average monthly electricity consumption
df['Month'] = df.index.to_period('M') # Convert datetime index to monthly periods
monthly_avg_kwh = df.groupby('Month')['kwh'].mean().reset_index()
monthly_avg_kwh['Month_Str'] = monthly_avg_kwh['Month'].astype(str)
plt.figure(figsize=(12, 7))
sns.barplot(x='Month_Str', y='kwh', data=monthly_avg_kwh, palette='magma')
plt.title('Average Monthly Electricity Consumption')
plt.xlabel('Month')
plt.ylabel('Average kWh Consumption')
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# Visualizing the average daily electricity consumption on holidays vs non-holidays
daily_avg = df.groupby(['date', 'Holiday'])['kwh'].mean().reset_index()
holidays = daily_avg[daily_avg['Holiday'] == 1]
non_holidays = daily_avg[daily_avg['Holiday'] == 0]
plt.figure(figsize=(14, 7))
plt.plot(holidays['date'], holidays['kwh'], label='Holiday', color='red', marker='o')
plt.plot(non_holidays['date'], non_holidays['kwh'], label='Non-Holiday', color='blue', marker='o')
plt.title('Average Daily Electricity Consumption: Holidays vs. Non-Holidays')
plt.xlabel('Date')
plt.ylabel('Average kWh Consumed')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizing the average electricity consumption by day of the week (non-holidays only)
non_holiday_df = df[df['Holiday'] == 0].copy()
non_holiday_df['dayofweek'] = non_holiday_df.index.dayofweek + 1
avg_by_day = non_holiday_df.groupby('dayofweek')['kwh'].mean().reset_index()
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.figure(figsize=(8, 5))
plt.bar(avg_by_day['dayofweek'], avg_by_day['kwh'], color='orange')
plt.title('Average Electricity Consumption by Day of Week (Non-Holidays Only)')
plt.xlabel('Day of Week')
plt.ylabel('Average kWh Consumed')
plt.xticks(ticks=range(1, 8), labels=day_labels)
plt.tight_layout()
plt.show()


# Visualizing the average daily electricity consumption on weekdays (Mon-Thu) for holidays vs non-holidays
df['dayofweek'] = df.index.dayofweek
df_weekdays = df[~df['dayofweek'].isin([4, 5, 6])].copy()
daily_avg = df_weekdays.groupby(['date', 'Holiday'])['kwh'].mean().reset_index()
holidays = daily_avg[daily_avg['Holiday'] == 1]
non_holidays = daily_avg[daily_avg['Holiday'] == 0]
plt.figure(figsize=(14, 7))
plt.plot(holidays['date'], holidays['kwh'], label='Holiday', color='red', marker='o')
plt.plot(non_holidays['date'], non_holidays['kwh'], label='Non-Holiday', color='blue', marker='o')
plt.title('Average Daily Electricity Consumption (Mon-Thu): Holidays vs. Non-Holidays')
plt.xlabel('Date')
plt.ylabel('Average kWh Consumed')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Visualizing the average electricity consumption for each 4-hour interval (non-holidays only)
non_holiday_df = df[df['Holiday'] == 0].copy()
non_holiday_df['4h_bin'] = (non_holiday_df.index.hour // 4) * 4
avg_4h = non_holiday_df.groupby('4h_bin')['kwh'].mean().reset_index()
labels = [f"{str(h).zfill(2)}:00-{str((h+4)%24).zfill(2)}:00" for h in avg_4h['4h_bin']]
plt.figure(figsize=(10, 5))
plt.plot(labels, avg_4h['kwh'], marker='o', color='green')
plt.title('Average Electricity Consumption for Each 4-Hour Interval (Non-Holidays)')
plt.xlabel('Time Interval')
plt.ylabel('Average kWh Consumed')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizing the average electricity consumption by temperature category
temp_bins = [-float('inf'), 0, 10, 20, float('inf')]
temp_labels = ['Very Cold', 'Cold', 'Normal', 'Hot']
df['temp_category'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False)
avg_kwh_by_temp = df.groupby('temp_category')['kwh'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='temp_category', y='kwh', data=avg_kwh_by_temp, palette='coolwarm')
plt.title('Average Electricity Consumption by Temperature Category')
plt.xlabel('Temperature Category')
plt.ylabel('Average kWh Consumption')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Model Training and Evaluation
final = final_csv.copy()
final['kwh'] = final['kwh'].apply(lambda x: max(0, x))

# --- Feature and Target Definition ---
TARGET = 'kwh'
# All features are now treated as numerical
FEATURES = ['day', 'temp_category_int', 'day_time', 'month',
            'Holiday', 'temperature', 'is_weekend']

# All features are now numerical, so no separate categorical list
numerical_features = FEATURES.copy() # All input features are numerical

# --- 1. Data Splitting (Chronological) ---
# Use a common split point for both models
train_size = int(len(final) * 0.7)
val_size = int(len(final) * 0.15)
test_size = len(final) - train_size - val_size

train_df = final.iloc[:train_size]
val_df = final.iloc[train_size:train_size + val_size]
test_df = final.iloc[train_size + val_size:]

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Train Start: {train_df.index.min()}, End: {train_df.index.max()}")
print(f"Val Start: {val_df.index.min()}, End: {val_df.index.max()}")
print(f"Test Start: {test_df.index.min()}, End: {test_df.index.max()}")

# --- Preprocessing Pipeline for Linear Regression ---
# Only numerical features now, so only StandardScaler
preprocessor_lr = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

print("\n--- Training Baseline Model (Linear Regression) ---")

# Separate features (X) and target (y)
X_train_lr = train_df[FEATURES]
y_train_lr = train_df[TARGET]
X_val_lr = val_df[FEATURES]
y_val_lr = val_df[TARGET]
X_test_lr = test_df[FEATURES]
y_test_lr = test_df[TARGET]

# Create a pipeline for preprocessing and model training
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('regressor', LinearRegression()) # Changed to LinearRegression
])

# Train the Linear Regression model
lr_pipeline.fit(X_train_lr, y_train_lr)

# Evaluate on validation set
y_val_pred_lr = lr_pipeline.predict(X_val_lr)
mae_val_lr = mean_absolute_error(y_val_lr, y_val_pred_lr)
rmse_val_lr = np.sqrt(mean_squared_error(y_val_lr, y_val_pred_lr))

print(f"Baseline Model (Linear Regression) Validation MAE: {mae_val_lr:.3f}")
print(f"Baseline Model (Linear Regression) Validation RMSE: {rmse_val_lr:.3f}")

# Evaluate on test set
y_test_pred_lr = lr_pipeline.predict(X_test_lr)
mae_test_lr = mean_absolute_error(y_test_lr, y_test_pred_lr)
rmse_test_lr = np.sqrt(mean_squared_error(y_test_lr, y_test_pred_lr))

print(f"Baseline Model (Linear Regression) Test MAE: {mae_test_lr:.3f}")
print(f"Baseline Model (Linear Regression) Test RMSE: {rmse_test_lr:.3f}")

print("\n--- Training LSTM Model (PyTorch Version) ---")

# Define sequence length (how many past timesteps to use for prediction)
LOOK_BACK = 24  # Use the past 24 hours of data to predict the next KWH

# Combine all features and target for scaling for LSTM
all_cols_for_lstm_scaling = FEATURES + [TARGET]

# Fit the scaler ONLY on the training data's relevant columns (all of them now)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[all_cols_for_lstm_scaling])

# Transform all datasets
train_scaled_data = scaler.transform(train_df[all_cols_for_lstm_scaling])
val_scaled_data = scaler.transform(val_df[all_cols_for_lstm_scaling])
test_scaled_data = scaler.transform(test_df[all_cols_for_lstm_scaling])

# Separate X and y from the scaled data for LSTM
target_col_index_in_scaled_data = all_cols_for_lstm_scaling.index(TARGET)

# Use numpy arrays directly, as create_sequences will handle it.
# We need to make sure X is 2D before passing to create_sequences.
X_train_lstm_full = np.delete(train_scaled_data, target_col_index_in_scaled_data, axis=1)
y_train_lstm_scaled = train_scaled_data[:, target_col_index_in_scaled_data]

X_val_lstm_full = np.delete(val_scaled_data, target_col_index_in_scaled_data, axis=1)
y_val_lstm_scaled = val_scaled_data[:, target_col_index_in_scaled_data]

X_test_lstm_full = np.delete(test_scaled_data, target_col_index_in_scaled_data, axis=1)
y_test_lstm_scaled = test_scaled_data[:, target_col_index_in_scaled_data]


# Function to create sequences for LSTM
# Modified to accept numpy arrays and return numpy arrays
def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

# Create sequences for training, validation, and testing
X_train_lstm, y_train_lstm = create_sequences(X_train_lstm_full, y_train_lstm_scaled, LOOK_BACK)
X_val_lstm, y_val_lstm = create_sequences(X_val_lstm_full, y_val_lstm_scaled, LOOK_BACK)
X_test_lstm, y_test_lstm = create_sequences(X_test_lstm_full, y_test_lstm_scaled, LOOK_BACK)

print(f"X_train_lstm shape: {X_train_lstm.shape}")
print(f"y_train_lstm shape: {y_train_lstm.shape}")

# Convert numpy arrays to PyTorch Tensors
# Ensure y is float32 and has a dimension for features (even if it's 1)
X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_lstm, dtype=torch.float32).unsqueeze(1) # Add feature dimension

X_val_tensor = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_lstm, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_lstm, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Build the LSTM model in PyTorch ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size) # Fully connected layer

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        # LSTM output: (output, (h_n, c_n))
        # output shape: (batch_size, sequence_length, hidden_size * num_directions)
        # We only care about the output from the last timestep for sequence prediction
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        # lstm_out[:, -1, :] gives the output of the last time step for all batches
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

# Model parameters
input_size = X_train_lstm.shape[2] # Number of features
hidden_size = 50 # Units in LSTM
output_size = 1 # Predicting KWH
dropout_rate = 0.2

model = LSTMModel(input_size, hidden_size, output_size, dropout_rate).to(device)

# Loss function and optimizer
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.Adam(model.parameters()) # Adam optimizer

print(model) # Print model architecture
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- Training Loop ---
NUM_EPOCHS = 50
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
history_train_loss = []
history_val_loss = []

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad() # Clear gradients
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    history_train_loss.append(avg_train_loss)

    # --- Validation ---
    model.eval() # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad(): # Disable gradient calculations for validation
        for batch_X_val, batch_y_val in val_loader:
            batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
            predictions_val = model(batch_X_val)
            loss_val = criterion(predictions_val, batch_y_val)
            val_loss += loss_val.item()

    avg_val_loss = val_loss / len(val_loader)
    history_val_loss.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

# Load the best model weights for evaluation
model.load_state_dict(torch.load('best_lstm_model.pth'))
print("Loaded best model weights.")

def evaluate_model(model, data_loader, scaler_obj, target_idx, original_scaled_data_for_inverse_transform, device):
    model.eval() # Set model to evaluation mode
    predictions_scaled = []
    actual_scaled = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions_scaled.extend(outputs.cpu().numpy().flatten())
            actual_scaled.extend(y_batch.cpu().numpy().flatten())

    # Inverse transform predictions and actual values
    # Need to reshape `actual_scaled` to (N, 1) to pass to inverse_transform_kwh
    y_pred_original = inverse_transform_kwh(np.array(predictions_scaled).reshape(-1, 1),
                                            original_scaled_data_for_inverse_transform,
                                            target_idx, scaler_obj)

    y_actual_original = inverse_transform_kwh(np.array(actual_scaled).reshape(-1, 1),
                                              original_scaled_data_for_inverse_transform,
                                              target_idx, scaler_obj)

    mae = mean_absolute_error(y_actual_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_actual_original, y_pred_original))
    return mae, rmse, y_pred_original, y_actual_original

# Function to inverse transform LSTM predictions (same as yours)
def inverse_transform_kwh(scaled_predictions, original_scaled_data, target_idx, scaler_obj):
    dummy_array = np.zeros((scaled_predictions.shape[0], original_scaled_data.shape[1]))
    dummy_array[:, target_idx] = scaled_predictions.flatten()
    original_values = scaler_obj.inverse_transform(dummy_array)
    return original_values[:, target_idx]

# Get the index of the TARGET column within the `all_cols_for_lstm_scaling` list
target_column_index_for_inverse_transform = all_cols_for_lstm_scaling.index(TARGET)

# Evaluate on Validation set
mae_val_lstm, rmse_val_lstm, y_val_pred_lstm, y_val_original = evaluate_model(
    model, val_loader, scaler, target_column_index_for_inverse_transform, val_scaled_data, device
)
print(f"\nLSTM Model Validation MAE: {mae_val_lstm:.3f}")
print(f"LSTM Model Validation RMSE: {rmse_val_lstm:.3f}")

# Evaluate on Test set
mae_test_lstm, rmse_test_lstm, y_test_pred_lstm, y_test_original = evaluate_model(
    model, test_loader, scaler, target_column_index_for_inverse_transform, test_scaled_data, device
)
print(f"LSTM Model Test MAE: {mae_test_lstm:.3f}")
print(f"LSTM Model Test RMSE: {rmse_test_lstm:.3f}")

print(f"Baseline (LR) Test MAE: {mae_test_lr:.3f}, RMSE: {rmse_test_lr:.3f}")
print(f"LSTM Test MAE: {mae_test_lstm:.3f}, RMSE: {rmse_test_lstm:.3f}")

LOOK_BACK = 24  # or your actual look-back value

plt.figure(figsize=(14, 7))
plt.plot(test_df.index, y_test_lr, label='Ground Truth', color='black')
plt.plot(test_df.index, y_test_pred_lr, label='LR', color='blue')
plt.plot(test_df.index[LOOK_BACK:], y_test_pred_lstm, label='LSTM (PyTorch)', color='red')
plt.title('Energy Consumption: Ground Truth vs. LR vs. LSTM (PyTorch)')
plt.xlabel('Time')
plt.ylabel('Energy Consumed (kWh)')
plt.legend()
plt.tight_layout()
plt.show()