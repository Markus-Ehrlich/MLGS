# script to fetch weather data for Lehnitz from https://open-meteo.com/en/docs/historical-weather-api#data_sources
# lat 52.732462 long 13.258927

import requests
from datetime import date, timedelta, datetime
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Ordner vorbereiten
data_dir_raw = pathlib.Path("data/raw")
data_dir_raw.mkdir(parents=True, exist_ok=True)

latitude = 52.732462
longitude = 13.258927

# Zeitfenster 
end_date = date.today()
start_date = end_date - timedelta(days=60)


# API abrufen
url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,sunset,sunrise,daylight_duration,sunshine_duration,rain_sum,snowfall_sum,precipitation_hours,weather_code,temperature_2m_mean,shortwave_radiation_sum,et0_fao_evapotranspiration"
)

response = requests.get(url)
data = response.json()

# In DataFrame umwandeln
df = pd.DataFrame({
    "date": data["daily"]["time"],
    "temp_max": data["daily"]["temperature_2m_max"],
    "temp_min": data["daily"]["temperature_2m_min"],
    "temp_mean": data["daily"]["temperature_2m_mean"],
    "wind_speed_max": data["daily"]["wind_speed_10m_max"],
    "wind_gusts_max": data["daily"]["wind_gusts_10m_max"],
    "wind_direction_dominant": data["daily"]["wind_direction_10m_dominant"],
    "sunset": data["daily"]["sunset"],
    "sunrise": data["daily"]["sunrise"],
    "daylight_duration": data["daily"]["daylight_duration"],
    "sunshine_duration": data["daily"]["sunshine_duration"],
    "rain_sum": data["daily"]["rain_sum"],
    "snowfall_sum": data["daily"]["snowfall_sum"],
    "precipitation_hours": data["daily"]["precipitation_hours"],
    "weather_code": data["daily"]["weather_code"],
    "shortwave_radiation_sum": data["daily"]["shortwave_radiation_sum"],
    "et0_fao_evapotranspiration": data["daily"]["et0_fao_evapotranspiration"]
    })

print(df.head())

# als json speichern
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
file_path = data_dir_raw / f"weather_data.json"
file_path.write_text(response.text, encoding="utf-8")

print(f"✅ Data fetched and saved to {file_path}")
