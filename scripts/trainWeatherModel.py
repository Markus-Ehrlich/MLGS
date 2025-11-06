from datetime import datetime
import pathlib
import subprocess
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Hardwarebeschleunigung nutzen, falls verfügbar
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dateipfade
data_dir = pathlib.Path("data/raw")
model_dir = pathlib.Path("models")
log_dir = pathlib.Path("logs")
log_file = pathlib.Path(log_dir / 'modelTrainingLog.csv')

# Daten laden
with open(data_dir / 'weather_data.json', 'r', encoding='utf-8') as f:
    weatherData = json.load(f)

# In DataFrame umwandeln
df = pd.DataFrame({
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

# Features (X) und Ziel (y) definieren
X = df.drop(columns=["date", "sunset", "sunrise", "weather_code", "temp_mean"])
y = df["temp_mean"]

# Datensatz aufteilen
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
# Ergebnis: 60% Train, 20% Valid, 20% Test

# In Torch-Tensoren umwandeln
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_valid = torch.tensor(y_valid.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Normalisierung
means = X_train.mean(dim=0, keepdims=True)
stds = X_train.std(dim=0, keepdims=True)
stds[stds == 0] = 1e-6  # Schutz gegen Division durch 0

X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

# Modellparameter
torch.manual_seed(42)
n_features = X_train.shape[1]
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Training
learning_rate = 0.01
n_epochs = 200

for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

# Modell speichern 
model_dir.mkdir(parents=True, exist_ok=True)
torch.save({'w': w, 'b': b, 'means': means, 'stds': stds}, model_dir / 'weatherModelParameters.pth')

# Evaluation
with torch.no_grad():
    y_pred_test = X_test @ w + b

# Metriken berechnen
y_true = y_test.numpy()
y_pred = y_pred_test.numpy()
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n📊 Modellbewertung:")
print(f"R² Score: {r2:.3f}  → Modellgüte: {r2 * 100:.1f}%")
print(f"MAE: {mae:.3f} °C (durchschnittlicher Fehler)")

# Git commit ID der Datenbasis ermitteln
try:
    commit_id = subprocess.check_output(['git','rev-parse','HEAD'], cwd=data_dir.parent).decode().strip()
except:
    commit_id = "unknown"

# Logging in CSV
log_exists = log_file.exists()
log_df = pd.DataFrame([{
    "date": datetime.now().strftime("%Y-%m-%d"),
    "time": datetime.now().strftime("%H:%M:%S"),
    "data_commit_id": commit_id,
    "r2_score": r2,
    "MAE": mae
}])
# no change
log_df.to_csv(log_file, mode='a', header=not log_exists, index=False)
