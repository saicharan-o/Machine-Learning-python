"""
Linear Regression - House Price Prediction
===========================================
Trains a simple linear regression model on house area (sqft) vs price data,
visualises the scatter plot, and makes price predictions for custom inputs.

Author : Sai Charan
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("house_price_single_linear_regression.csv")

# ── Visualise Raw Data ────────────────────────────────────────────────────────
plt.ticklabel_format(style='plain', axis='y')
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("House Area vs Price")
plt.scatter(df.Area_sqft, df.Price, color="r", marker=".")
plt.show()

# ── Train Model ───────────────────────────────────────────────────────────────
X = df[["Area_sqft"]]
y = df[["Price"]]

model = LinearRegression()
model.fit(X, y)

print("Slope     :", model.coef_)
print("Intercept :", model.intercept_)

# ── Predictions ───────────────────────────────────────────────────────────────
print("Prediction 1 (2030 sqft) :", model.predict(pd.DataFrame({"Area_sqft": [2030]})))
print("Prediction 2 (48202 sqft):", model.predict(pd.DataFrame({"Area_sqft": [48202]})))
