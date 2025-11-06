import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# Generate dummy dataset (X: 5 features, y: 1 target)
X = np.random.rand(200, 5) * 100
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 5 + np.random.randn(200) * 5

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Save both model and scaler
joblib.dump(model, "models/linear_regression.joblib")
joblib.dump(scaler, "models/standard_scaler.joblib")

print("✅ linear_regression.joblib and standard_scaler.joblib created successfully.")
