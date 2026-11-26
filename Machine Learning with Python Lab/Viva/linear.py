
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load dataset
train = pd.read_csv("california_housing_train.csv")
test = pd.read_csv("california_housing_test.csv")

# 2. Separate features and target
X_train = train.drop("median_house_value", axis=1).values
y_train = train["median_house_value"].values

X_test = test.drop("median_house_value", axis=1).values
y_test = test["median_house_value"].values

# 3. Feature scaling
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 4. Add bias column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# 5. Normal Equation
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

pred1 = X_test @ theta

# 6. Batch Gradient Descent
theta_gd = np.zeros(X_train.shape[1])
lr = 0.01
n = len(y_train)

for i in range(1000):
    error = X_train @ theta_gd - y_train
    gradient = (X_train.T @ error) / n
    theta_gd = theta_gd - lr * gradient

pred2 = X_test @ theta_gd

# 7. Scikit-learn Linear Regression
model = LinearRegression()
model.fit(X_train[:, 1:], y_train)

pred3 = model.predict(X_test[:, 1:])

# 8. Evaluation Metrics
def evaluate(y, pred):
    mse = np.mean((y - pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - pred))
    r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)

print("Normal Equation:")
evaluate(y_test, pred1)

print("Gradient Descent:")
evaluate(y_test, pred2)

print("Scikit-learn:")
evaluate(y_test, pred3)
