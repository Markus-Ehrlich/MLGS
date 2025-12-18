"""Process fetched weather data to make it ready for model training."""

import pathlib
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


import matplotlib.pyplot as plt

# Paths definition
DATA_DIR_RAW = pathlib.Path("data/raw")
DATA_DIR_PROCESSED = pathlib.Path("data/processed")
DATA_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)


# Open raw data file
with open(DATA_DIR_RAW / "weather_data.json", "r", encoding="utf-8") as f:
    weatherData = json.load(f)

# Transform feedback from API into DataFrame
df_raw = pd.DataFrame(
    {
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
        "shortwave_radiation_sum": weatherData["daily"][
            "shortwave_radiation_sum"
        ],  # How much solar radiation reached the ground
        "et0_fao_evapotranspiration": weatherData["daily"][
            "et0_fao_evapotranspiration"
        ],  # How much water would evaporate, given sufficient water supply
    }
)

# Print DataFrame description for verification
print("Raw feature data:")
print(df_raw.head())
print(df_raw.info())
print(df_raw.describe())
print(df_raw.isnull().sum())
print(len(df_raw["temp_max"]))

# Clean the data
# Create intermediate features dataframe
# All numerical features
df_features_num: pd.DataFrame = df_raw.select_dtypes(include=["number"]).copy()
# All object features
df_features_obj: pd.DataFrame = df_raw.select_dtypes(exclude=["number"]).copy()
# Drop date column
df_features_obj.drop(columns=["date"]).copy()

# Impute missing values
# Numerical features: fill with median
imputer = SimpleImputer(strategy="median")
imputer.fit(df_features_num)
X = imputer.transform(df_features_num)
df_features_num_imputed = pd.DataFrame(
    X, columns=df_features_num.columns, index=df_features_num.index
)

# Object features: fill with 'unknown'
imputer_obj = SimpleImputer(strategy="constant", fill_value="unknown")
imputer_obj.fit(df_features_obj)
X_obj = imputer_obj.transform(df_features_obj)
df_features_obj_imputed = pd.DataFrame(
    X_obj, columns=df_features_obj.columns, index=df_features_obj.index
)

# Process features physicallywise
# Numerical features
# Create combined precipitation sum feature and drop individual ones
df_features_num_imputed["precipitation_sum"] = (
    df_features_num_imputed["rain_sum"] + df_features_num_imputed["snowfall_sum"]
)
df_features_num_imputed.drop(columns=["rain_sum", "snowfall_sum"], inplace=True)

# Weather code: one-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_weather_code = encoder.fit_transform(
    df_features_num_imputed[["weather_code"]]
)
df_encoded_weather_code = pd.DataFrame(
    encoded_weather_code, columns=encoder.get_feature_names_out(
        df_features_num_imputed[["weather_code"]].columns
    )
)

# Append encoded weather code features and drop original column
df_features_num_imputed = pd.concat(
    [df_features_num_imputed, df_encoded_weather_code], axis=1
)
df_features_num_imputed.drop(columns=["weather_code"], inplace=True)

# Object features
# drop date
df_features_obj_imputed.drop(columns=["date"], inplace=True)
# temporary workaround: Object values must be dropped until they have been processes properly
df_features_obj_imputed.drop(columns=["sunset", "sunrise"], inplace=True) 

# Scaling
# precipitation_sum
standard_scaler = StandardScaler()
df_features_num_imputed["precipitation_sum_scaled"] = standard_scaler.fit_transform(
    df_features_num_imputed[["precipitation_sum"]]
)
df_features_num_imputed.drop(columns=["precipitation_sum"])

# Placeholder for value conditioning, normalization etc.

# ##################################
# ##################################

# Combine processed numerical and object features
df_features = pd.concat([df_features_num_imputed, df_features_obj_imputed], axis=1)

# Features for the last available day to predict next day
# I am not so sure yet, if this will be needed
df_features_for_tomorrow = df_features.iloc[[-1]]  # not used yet

# Delete last row from features dataframe to align with target
df_features = df_features.drop(df_features.index[-1])

# Select target (next day's max temperature)
df_target = df_raw["temp_max"].shift(-1).to_frame()
df_target = df_target.dropna()



# Print overview of processed data
print("Processed feature data:")
print(df_features.head())
print(df_features.info())
print(df_features_for_tomorrow.head())
print(df_features_for_tomorrow.info())
print("Processed target data:")
print(df_target.head())
print(df_target.info())

# Save processed data
processed_data = {
    "features": df_features.to_dict(orient="split"),
    "target": df_target.to_dict(orient="split"),
    "features_for_tomorrow": df_features_for_tomorrow.to_dict(orient="split"),
}
with open(
    DATA_DIR_PROCESSED / "weather_data_processed.json", "w", encoding="utf-8"
) as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
