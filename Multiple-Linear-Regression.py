"""
Multiple Linear Regression - House Price Prediction
====================================================
Trains a multiple linear regression model on area, bedrooms, and bathrooms
to predict house prices, generates four diagnostic scatter plots, and makes
price predictions for custom property configurations.

Author : Sai Charan
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("House_price_multiple_linear_Regression.csv")

X = df.drop("Price", axis=1)
y = df.Price

# ── Train Model ───────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X, y)

print("Slope     :", model.coef_)
print("Intercept :", model.intercept_)

# ── Predictions ───────────────────────────────────────────────────────────────
print("Prediction 1 (3040 sqft, 3 bed, 2 bath):",
      model.predict(pd.DataFrame({"Area_sqft": [3040], "Bedrooms": [3], "Bathrooms": [2]})))
print("Prediction 2 (300 sqft, 1 bed, 1 bath):",
      model.predict(pd.DataFrame({"Area_sqft": [300], "Bedrooms": [1], "Bathrooms": [1]})))
print("Prediction 3 (5400 sqft, 5 bed, 3 bath):",
      model.predict(pd.DataFrame({"Area_sqft": [5400], "Bedrooms": [5], "Bathrooms": [3]})))

# ── Diagnostic Scatter Plots ──────────────────────────────────────────────────
plots = [
    ("Area_sqft", "Area (sqft)", "Price"),
    ("Bedrooms",  "Bedrooms",    "Price"),
    ("Bathrooms", "Bathrooms",   "Price"),
]

for col, xlabel, ylabel in plots:
    plt.scatter(df[col], df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}")
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

# ── Actual vs Predicted ───────────────────────────────────────────────────────
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.ticklabel_format(style='plain', axis='both')
plt.show()