"""Process fetched weather data to make it ready for model training."""

#from datetime import date, timedelta, datetime
import pathlib
#import requests
import json
import pandas as pd

# Paths definition
DATA_DIR_RAW = pathlib.Path("data/raw")
DATA_DIR_PROCESSED = pathlib.Path("data/processed")
DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

# Daten laden
with open(DATA_DIR_RAW / 'weather_data.json', 'r', encoding='utf-8') as f:
    weatherData = json.load(f)

# Transform feedback from API into DataFrame
df_raw = pd.DataFrame({
    "date": weatherData["daily"]["time"],
    "temp_max": weatherData["daily"]["temperature_2m_max"],
    "temp_min": weatherData["daily"]["temperature_2m_min"],
    "temp_mean": weatherData["daily"]["temperature_2m_mean"],
    "wind_speed_max": weatherData["daily"]["wind_speed_10m_max"],
    "wind_gusts_max": weatherData["daily"]["wind_gusts_10m_max"],
    "wind_direction_dominant": weatherData["daily"]["wind_direction_10m_dominant"],
    "sunset": weatherData["daily"]["sunset"],
    "sunrise": weatherData["daily"]["sunrise"],
    "daylight_duration": weatherData["daily"]["daylight_duration"],
    "sunshine_duration": weatherData["daily"]["sunshine_duration"],
    "rain_sum": weatherData["daily"]["rain_sum"],
    "snowfall_sum": weatherData["daily"]["snowfall_sum"],
    "precipitation_hours": weatherData["daily"]["precipitation_hours"],
    "weather_code": weatherData["daily"]["weather_code"],
    "shortwave_radiation_sum": weatherData["daily"]["shortwave_radiation_sum"],
    "et0_fao_evapotranspiration": weatherData["daily"]["et0_fao_evapotranspiration"]
})

# Print head of DataFrame for verification
print(df_raw.head())

# Placeholder for value conditioning, normalization etc.

# ##################################
# ##################################

# Split data frame into features and target
# Drop date and target
df_features = df_raw.drop(columns=["date", "temp_mean"])

# Features for the last available day to predict next day
df_features_for_tomorrow = df_features.iloc[[-1]]

# Delete last row from features dataframe to align with target
df_features = df_features.drop(df_features.index[-1])

# Target is next day's mean temperature
df_target = df_raw["temp_mean"].shift(-1).to_frame()
df_target = df_target.dropna()

# Save processed data
processed_data = {
    "features": df_features.to_dict(orient="split"),
    "target": df_target.to_dict(orient="split"),
    "features_for_tomorrow": df_features_for_tomorrow.to_dict(orient="split")
}

with open(DATA_DIR_PROCESSED / 'weather_data_processed.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)