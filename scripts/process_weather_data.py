"""Process fetched weather data to make it ready for model training."""

import pathlib
import json
import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Paths definition
DATA_DIR_RAW = pathlib.Path("data/raw")
DATA_DIR_PROCESSED = pathlib.Path("data/processed")
DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)


# Open raw data file
with open(DATA_DIR_RAW / 'weather_data.json', 'r', encoding='utf-8') as f:
    weatherData = json.load(f)

# %%
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
    "shortwave_radiation_sum": weatherData["daily"]["shortwave_radiation_sum"], # How much solar radiation reached the ground
    "et0_fao_evapotranspiration": weatherData["daily"]["et0_fao_evapotranspiration"] # How much water would evaporate, given sufficient water supply
})

# %%
# Print head of DataFrame for verification
print(df_raw.head())

# Collect features
# Just drop date
df_features: pd.DataFrame = df_raw.drop(columns=["date"]).copy()

# %%
# Process features physicallywise
# Create combined precipitation sum feature and drop individual ones
df_features["precipitation_sum"] = df_features["rain_sum"] + df_features["snowfall_sum"]
df_features.drop(columns=["rain_sum", "snowfall_sum"], inplace=True)

# temporary workaround: Numerical values must be dropped until they have been processes properly
df_features = df_features.drop(columns=["sunset", "sunrise", "weather_code"])

# %%
# Features for the last available day to predict next day
# I am not so sure yet, if this will be needed
df_features_for_tomorrow = df_features.iloc[[-1]] # not used yet

# Delete last row from features dataframe to align with target
df_features = df_features.drop(df_features.index[-1])

# %%
# Select target (next day's max temperature)
df_target = df_raw["temp_max"].shift(-1).to_frame()
df_target = df_target.dropna()

# Processing for model readability
# precipitation_sum

#fillna
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(strategy="median")

# %%
standard_scaler = StandardScaler()
df_features["precipitation_sum_scaled"] = standard_scaler.fit_transform(
    df_features[["precipitation_sum"]])
df_features = df_features.drop(columns=["precipitation_sum"])

# Placeholder for value conditioning, normalization etc.

# ##################################
# ##################################

# %%
# Print overview of processed data
print("Processed feature data:")
print(df_features.head())
print(df_features.info())
print(df_features_for_tomorrow.head())
print(df_features_for_tomorrow.info())
print("Processed target data:")
print(df_target.head())
print(df_target.info())

# %%
# Save processed data
processed_data = {
    "features": df_features.to_dict(orient="split"),
    "target": df_target.to_dict(orient="split"),
    "features_for_tomorrow": df_features_for_tomorrow.to_dict(orient="split")
}

with open(DATA_DIR_PROCESSED / 'weather_data_processed.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)