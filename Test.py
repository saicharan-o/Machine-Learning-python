import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.ticklabel_format(style='plain', axis='y')
d=pd.read_csv("H-P-P.csv")
x=d.drop("Price",axis=1)
y=d.Price
md=LinearRegression()
md.fit(x,y)
print("Slope: \n", md.coef_)
print("Intercept: \n", md.intercept_)
print("Linear Regression-Prediction 1: ",md.predict(pd.DataFrame({"Area_sqft":[3040],"Bedrooms":[3],"Bathrooms":[2]})))
print("Linear Regression-Prediction 2: ",md.predict(pd.DataFrame({"Area_sqft":[300],"Bedrooms":[1],"Bathrooms":[1]})))
print("Linear Regression-Prediction 3: ",md.predict(pd.DataFrame({"Area_sqft":[5400],"Bedrooms":[5],"Bathrooms":[3]})))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x=d.drop("Price",axis=1)
y=d.Price
m=LinearRegression()
X_tr,X_t,y_tr,y_t=train_test_split(x,y,test_size=0.3)
m.fit(X_tr,y_tr)
print(m.predict(X_t))
print("Score: \n",m.score(X_t,y_t))
