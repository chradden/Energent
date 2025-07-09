import pandas as pd
import os

class PVForecaster:
    def __init__(self):
        self.data = []
        self.model = None
        
    def read_data(self, path_to_pv_data):
        df = pd.read_csv(path_to_pv_data, sep=";")
        df["date_time"] = pd.to_datetime(df["date_time"], format="%d.%m.%Y %H:%M")
        df.set_index("date_time", inplace=True)
        # df["pv_electricity(kW)"] = pd.to_numeric(df["pv_electricity(kW)"])
        # df = df.resample('h').mean()
        self.data = df
        return df
    
    def get_weather_data(self, url, location):
        # This function should fetch weather data from a weather API based on the url and location.
        pass


if __name__ == "__main__":
    pass