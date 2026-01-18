import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
d=pd.read_csv("House_price_multiple_linear_Regression.csv")
x=d.drop(d.Price,axis="columns")
y=d.Price
md=LinearRegression()
md.fit(x,y)
print("Slope: \n", md.coef_)
print("Intercept: \n", md.intercept_)
print("Prediction 1: ",md.predict(pd.DataFrame({"Area_sqft":[3040],"Bedrooms":[3],"Bathrooms":[2]})))
print("Prediction 2: ",md.predict(pd.DataFrame({"Area_sqft":[300],"Bedrooms":[1],"Bathrooms":[1]})))
print("Prediction 3: ",md.predict(pd.DataFrame({"Area_sqft":[5400],"Bedrooms":[5],"Bathrooms":[3]})))
