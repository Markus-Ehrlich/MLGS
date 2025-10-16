import sklearn
import torch
import json
import pathlib
from sklearn.model_selection import train_test_split

# Hardwarebeschleunigung
device = "cuda"

# Daten laden 
data_dir = pathlib.Path("data/raw")
with open(data_dir / 'weather_data.json', 'r', encoding='utf-8') as f:
    weatherData = json.load(f)

# Daten aufteilen
#weatherData = d
X_train_full, X_test, y_train_full, y_test = train_test_split(
    weatherData["daily"]["temperature_2m_min"], weatherData["daily"]["temperature_2m_max"], random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)


#
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
means = X_train.mean(dim=0, keepdims=True)
stds = X_train.std(dim=0, keepdims=True)
X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds


y_train = torch.FloatTensor(y_train).view(-1, 1)
y_valid = torch.FloatTensor(y_valid).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

torch.manual_seed(42)
n_features = X_train.shape[0]  # there are 8 input features
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.4
n_epochs = 20
for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        b -= learning_rate * b.grad
        w -= learning_rate * w.grad
        b.grad.zero_()
        w.grad.zero_()
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")