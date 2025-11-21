"""Train and evaluate a linear regression model for weather data."""

from datetime import datetime
import pathlib
import subprocess
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Use GPU if available
TRAINING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths definition
data_dir = pathlib.Path("data/processed")
model_dir = pathlib.Path("models")
log_dir = pathlib.Path("logs")
log_file = pathlib.Path(log_dir / 'modelTrainingLog.csv')

# Load preprocessed data
with open(data_dir / 'weather_data_processed.json', 'r', encoding='utf-8') as f:
    weatherData = json.load(f)

# Define features and target
X = pd.DataFrame(
    index = weatherData["features"]["index"],
    columns = weatherData["features"]["columns"],
    data = weatherData["features"]["data"]
)
y = pd.DataFrame(
    index = weatherData["target"]["index"],
    columns = weatherData["target"]["columns"],
    data = weatherData["target"]["data"]
)

# Split data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25,
                                                       random_state=42)
# Result: 60% Train, 20% Valid, 20% Test

# Convert to torch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_valid = torch.tensor(y_valid.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Normalization
means = X_train.mean(dim=0, keepdims=True)
stds = X_train.std(dim=0, keepdims=True)
stds[stds == 0] = 1e-6  # Secure against division by zero

X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds

# Model parameters initialization
torch.manual_seed(42)
n_features = X_train.shape[1]
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Model training
LEARNING_RATE = 0.01
N_EPOCHS = 200

for epoch in range(N_EPOCHS):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= LEARNING_RATE * w.grad
        b -= LEARNING_RATE * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {loss.item():.4f}")

# Save model parameters 
model_dir.mkdir(parents=True, exist_ok=True)
torch.save({'w': w, 'b': b, 'means': means, 'stds': stds}, model_dir / 'weatherModelParameters.pth')

# Evaluate model on test set
with torch.no_grad():
    y_pred_test = X_test @ w + b

# Calculate performance metrics
y_true = y_test.numpy()
y_pred = y_pred_test.numpy()
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n Model evaluation:")
print(f"R² Score: {r2:.3f}  → Model quality: {r2 * 100:.1f}%")
print(f"MAE: {mae:.3f} °C (average prediction error)")

# Get git commit ID of data basis
try: # verify this is the correct approach; ID of raw data commit is needed
    COMMIT_ID = subprocess.check_output(['git','rev-parse','HEAD'],
                                         cwd=data_dir.parent).decode().strip()
except ImportError:
    COMMIT_ID = "unknown"

# Save log entry
LOG_EXISTS = log_file.exists()
log_df = pd.DataFrame([{
    "date": datetime.now().strftime("%Y-%m-%d"),
    "time": datetime.now().strftime("%H:%M:%S"),
    "data_commit_id": COMMIT_ID,
    "r2_score": r2,
    "MAE": mae
}])
# no change
log_df.to_csv(log_file, mode='a', header=not LOG_EXISTS, index=False)
