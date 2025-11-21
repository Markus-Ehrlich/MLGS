"""script to fetch weather data for Lehnitz from 
https://open-meteo.com/en/docs/historical-weather-api#data_sources
lat 52.732462 long 13.258927"""

from datetime import date, timedelta, datetime
import pathlib
import requests

# Prepare folder
DATA_DIR_RAW = pathlib.Path("data/raw")
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)

LATITUDE = 52.732462
LONGITUDE = 13.258927

# Define time range
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=60)


# API call
APIURL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,"
    "wind_direction_10m_dominant,sunset,sunrise,daylight_duration,sunshine_duration,rain_sum,"
    "snowfall_sum,precipitation_hours,weather_code,temperature_2m_mean,shortwave_radiation_sum,"
    "et0_fao_evapotranspiration"
)
response = requests.get(APIURL, timeout=10)

# Save raw JSON data
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
file_path = DATA_DIR_RAW / "weather_data.json"
file_path.write_text(response.text, encoding="utf-8")

print(f"Data fetched and saved to {file_path}")
