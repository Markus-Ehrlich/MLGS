# script to fetch weather data for Lehnitz from https://open-meteo.com/en/docs/historical-weather-api#data_sources
# Breite 52.732462 Länge 13.258927

import requests
from datetime import date, timedelta, datetime
import pathlib
import pandas as pd

# Ordner vorbereiten
data_dir = pathlib.Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

latitude = 48.14
longitude = 11.58

# Zeitfenster (letzte 7 Tage)
end_date = date.today()
start_date = end_date - timedelta(days=7)

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
    "&timezone=Europe/Berlin"
)

response = requests.get(url)
data = response.json()

# In DataFrame umwandeln
df = pd.DataFrame({
    "date": data["daily"]["time"],
    "temp_max": data["daily"]["temperature_2m_max"],
    "temp_min": data["daily"]["temperature_2m_min"],
    "precipitation": data["daily"]["precipitation_sum"]
})

print(df.head())




timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
file_path = data_dir / f"data_{timestamp}.json"
file_path.write_text(response.text, encoding="utf-8")

print(f"✅ Data fetched and saved to {file_path}")
