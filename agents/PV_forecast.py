import polars as pl 
import requests
import os
from suncalc import get_position
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta  
import numpy as np

class PVForecaster:
    def __init__(self, latitude: float, longitude: float):
        self.data = []
        self.model = None
        self.model_mse = None
        self.latitude = latitude
        self.longitude = longitude
        self.weather_variables = ["direct_radiation", "cloud_cover"]
        self.timezone="Europe/Berlin"
        
    def prepare_data(self, path_to_pv_data: str):
        ## run this only once to prepare data
        ## hardcoded for the specific file for the moment
        df = pl.read_csv(path_to_pv_data, skip_lines=3, has_header=False, separator=";", new_columns=["date_time", "pv_electricity(kW)"], decimal_comma=True)
        ## convert date_time to datetime
        df = df.with_columns(
            pl.col("date_time").str.to_datetime("%d.%m.%Y %H:%M")
        )
        ## downsample to hourly data
        df = df.sort("date_time")
        df = df.group_by_dynamic("date_time", every="1h").agg(pl.col("pv_electricity(kW)").mean())
        ## get sun position for each hour and add to the dataframe
        df = df.with_columns(
            pl.col("date_time").map_elements(lambda x: get_position(x, lng=self.longitude, lat=self.latitude)).alias("sun_position")
        )
        df = df.with_columns(
            pl.col("sun_position").struct.unnest()
        )
        df = df.drop(["sun_position"])
        ## add our and week
        df = df.with_columns(
            pl.col("date_time").dt.week().alias("week"),
            pl.col("date_time").dt.hour().alias("hour")
        )
        ## get weather data from open-meteo
        start_date = df["date_time"].to_list()[0]
        end_date = df["date_time"].to_list()[-1]
        weather = self.get_houly_weather(self.latitude, self.longitude, start_date, end_date, self.weather_variables)
        weather_df = pl.from_dict(weather)
        weather_df = weather_df.with_columns(pl.col("time").str.to_datetime())
        ## merge df with weather data
        df = df.join(weather_df, left_on="date_time", right_on="time")
        ## save to csv
        folder = os.path.dirname(path_to_pv_data)
        df.write_csv(f"{folder}/PV-prepared.csv")
        
            
    def get_houly_weather(self, lat: float, lon: float, start_date, end_date, variables: list):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "timezone": self.timezone
        }
        response = requests.get(url, params=params)
        data = response.json()
        weather = dict()
        for var in data["hourly"].keys():
            weather[var] = data["hourly"][var]
        return weather
        
    def read_prepared_data(self, path_to_data_file):
        # test for prepared data file, create when not there
        if not os.path.exists(path_to_data_file):
            print("no prepared data set found. use .prepare_data(original_data_file_path) method to prepare the data set.")
        else:
            self.data = pl.read_csv(path_to_data_file)
            
    def train_xgboost(self):
        # XGBoost model training logic here
        
        # check if data loaded
        if not isinstance(self.data, pl.DataFrame):
            print("No data available for training.")
            return None
        
        # Prepare the dataset for XGBoost
        cols = self.data.columns
        cols.remove("date_time")
        cols.remove("pv_electricity(kW)")
        features = self.data[cols]
        target = self.data["pv_electricity(kW)"]
        print("[DEBUG] Trainings-Features (X):")
        print(features.head(24))
        print("[DEBUG] Zielwerte (y):")
        print(target.head(24))
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
        
        # train model
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        self.model = xgb_model
        
        # evaluate
        y_pred = xgb_model.predict(X_test) 
        mse = mean_squared_error(y_test, y_pred)
        self.model_mse = mse
        print(f"[DEBUG] Modellgüte (MSE): {mse}")
    
    def predict_next_day(self):
        # get weather and other stuff
        weather = self.weather_forecast(self.latitude, self.longitude, self.weather_variables)
        df = pl.from_dict(weather)
        df = df.with_columns(pl.col("time").str.to_datetime())
        ## get sun position for each hour and add to the dataframe
        df = df.with_columns(
            pl.col("time").map_elements(lambda x: get_position(x, lng=self.longitude, lat=self.latitude)).alias("sun_position")
        )
        df = df.with_columns(
            pl.col("sun_position").struct.unnest()
        )
        df = df.drop(["sun_position"])
        ## add our and week
        df = df.with_columns(
            pl.col("time").dt.week().alias("week"),
            pl.col("time").dt.hour().alias("hour")
        )
        df = df.drop("time")
        # reorder
        df = df.select(
            pl.col("azimuth"),
            pl.col("altitude"),
            pl.col("week"),
            pl.col("hour"),
            pl.col("direct_radiation"),
            pl.col("cloud_cover")
        )
        print("[DEBUG] Features für Vorhersage (next day):")
        print(df.head(24))
        y_pred = self.model.predict(df)
        print("[DEBUG] Vorhersage (y_pred):")
        print(y_pred[:24])
        return y_pred
    
    def weather_forecast(self, lat: float, lon: float, variables: list):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "forecast_days": 1,
            "timezone": self.timezone
        }
        response = requests.get(url, params=params)
        data = response.json()
        weather = dict()
        for var in data["hourly"].keys():
            weather[var] = data["hourly"][var]
        return weather



if __name__ == "__main__":
    lat= 53.5511
    lon= 9.9937
    orig_file_path = "data/historical_Data/PV-electricity_2024_01_01.csv"
    folder = os.path.dirname(orig_file_path)
    prep_file_path = folder + "/PV-prepared.csv"
    
    pv_forecast = PVForecaster(latitude=lat, longitude=lon)
    
    # test for prepared data file, create when not there
    folder = os.path.dirname(orig_file_path)
    if not os.path.exists(prep_file_path):
        print("preparing data set")
        pv_forecast.prepare_data(orig_file_path)
    else:
        print("found prepared data. continuing...")
    # read data file
    pv_forecast.read_prepared_data(prep_file_path)
    # train xgb model
    pv_forecast.train_xgboost()
    print("XGBoost model trained successfully.")
    print("Model evaluation:")
    print(f"mse: {pv_forecast.model_mse}")
    print(pv_forecast.predict_next_day())
    