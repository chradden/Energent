import polars as pl 
import requests
import os
from suncalc import get_position

class PVForecaster:
    def __init__(self, latitude: float, longitude: float):
        self.data = []
        self.model = None
        self.latitude = latitude
        self.longitude = longitude
        
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
        variables = ["direct_radiation", "cloud_cover"]
        weather = self.get_houly_weather(self.latitude, self.longitude, start_date, end_date, variables)
        weather_df = pl.from_dict(weather)
        weather_df = weather_df.with_columns(pl.col("time").str.to_datetime())
        ## merge df with weather data
        df = df.join(weather_df, left_on="date_time", right_on="time")
        ## save to csv
        folder = os.path.dirname(path_to_pv_data)
        df.write_csv(f"{folder}/PV-prepared.csv")
        
            
    def get_houly_weather(self, lat: float, lon: float, start_date, end_date, variables: list, timezone="Europe/Berlin"):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "timezone": timezone
        }
        response = requests.get(url, params=params)
        data = response.json()
        weather = dict()
        for var in data["hourly"].keys():
            weather[var] = data["hourly"][var]
        return weather
        
    # def read_data(self, path_to_pv_data):
    #     df = pd.read_csv(path_to_pv_data, sep=";")
    #     df["date_time"] = pd.to_datetime(df["date_time"], format="%d.%m.%Y %H:%M")
    #     df.set_index("date_time", inplace=True)
    #     # df["pv_electricity(kW)"] = pd.to_numeric(df["pv_electricity(kW)"])
    #     # df = df.resample('h').mean()
    #     self.data = df
    #     return df
    


if __name__ == "__main__":
    lat= 53.5511
    lon= 9.9937
    orig_file_path = "data/historical_Data/PV-electricity_2024_01_01.csv"
    
    pv_forecast = PVForecaster(latitude=lat, longitude=lon)
    
    # test for prepared data file, create when not there
    folder = os.path.dirname(orig_file_path)
    if not os.path.exists(f"{folder}/PV-prepared.csv"):
        print("preparing data set")
        df = pv_forecast.prepare_data(orig_file_path)
    else:
        print("found prepared data. continuing...")