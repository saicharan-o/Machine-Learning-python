import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.ticklabel_format(style='plain', axis='y')
d=pd.read_csv("house_price_single_linear_regression.csv")
plt.xlabel("Area_sqft")
plt.ylabel("Price")
plt.scatter(d.Area_sqft,d.Price,color="r",marker=".")
plt.show()
x=d[["Area_sqft"]]
y=d[["Price"]]
md=LinearRegression()
md.fit(x,y)
print("Slope: \n", md.coef_)
print("Intercept: \n", md.intercept_)
print("Prediction 1: ",md.predict(pd.DataFrame({"Area_sqft":[2030]})))
print("Prediction 2: ",md.predict(pd.DataFrame({"Area_sqft":[48202]})))
