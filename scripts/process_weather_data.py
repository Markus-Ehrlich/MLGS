"""Process fetched weather data to make it ready for model training.

This script performs the following steps:
1. Loads raw weather data from a JSON file.
2. Imputes missing values for numerical and categorical features.
3. Creates new features (e.g., combined precipitation sum, one-hot encoding of weather code).
4. Scales and conditions numerical features (e.g., log transformation and standard scaling
of precipitation sum).
5. Formats the data into a structured format suitable for model training, including 
separating features and target variable.
6. Saves the processed data to a new JSON file for use in model training.
"""

import pathlib
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

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

# Feature engineering
df_processing = df_raw.copy()
# Create combined precipitation sum feature and drop individual ones
df_processing["precipitation_sum"] = (
    df_processing["rain_sum"] + df_processing["snowfall_sum"]
)
# Drop input values
df_processing.drop(columns=["rain_sum", "snowfall_sum"], inplace=True)

# Convert wind direction to sine and cosine components
df_processing["wind_dir_sin"] = np.sin(np.deg2rad(
    df_processing["wind_direction_dominant"]))
df_processing["wind_dir_cos"] = np.cos(np.deg2rad(
    df_processing["wind_direction_dominant"]))
# Drop input values
df_processing.drop(columns=["wind_direction_dominant"], inplace=True)

# Drop unused attributes for now, as they are not useful (yet)
df_processing.drop(columns=["date", "sunset", "sunrise"], inplace=True)

# Verify columns in df_raw before applying the pipeline
print("Columns in df_raw before pipeline:")
print(df_processing.columns)

# Define attribute categories for pipeline processing
# Right now, no categorical attributes are used
# Numerical attributes that require log transformation due to skewness
log_attributes = ["et0_fao_evapotranspiration", "precipitation_sum"]

# Categorical attributes that require one-hot encoding (it is officially numerical)
onehot_attributes = ["weather_code"]

# Sort leftover attributes input into generic numeric group
all_numeric = df_processing.select_dtypes(include=["number"]).columns.tolist()
# already used numeric attributes in the lists above
used_numeric = (
    log_attributes
    + onehot_attributes
)
# Only list numerical attributes that are not already used in the above pipelines
num_attributes = [col for col in all_numeric if col not in used_numeric]

# Pipeline for log transformation of skewed features: imputation, log transformation, and scaling
log_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # Handle missing values
    ("log_transform", FunctionTransformer(np.log1p, validate=False, 
                                          feature_names_out="one-to-one")),  # Log transformation
    ("scale", StandardScaler())  # Standard scaling
])

# Standard pipeline for numerical features: imputation with median and standard scaling
numerical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
])

# Pipeline for categorical features: imputation with 'unknown' and one-hot encoding
onehot_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# Combine pipelines into a ColumnTransformer
preprocessing_pipeline = ColumnTransformer([
    ("log_transform", log_pipeline, log_attributes),
    ("cat", onehot_pipeline, onehot_attributes),
    ("num", numerical_pipeline, num_attributes)
])

# Carry out the preprocessing pipeline on the raw data
df_preprocessed = preprocessing_pipeline.fit_transform(df_processing)
# Transform the preprocessed data back into a DataFrame with appropriate column names
feature_names = preprocessing_pipeline.get_feature_names_out()
df_preprocessed = pd.DataFrame(
    df_preprocessed,
    columns=feature_names,
    index=df_processing.index
)

# Features for the last available day to predict next day
# I am not so sure yet, if this will be needed
df_features_for_tomorrow = df_preprocessed.iloc[[-1]]  # not used yet

# Delete last row from features dataframe to align with target
df_features = df_preprocessed.drop(df_preprocessed.index[-1])

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
