import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("Hours.csv")
print(d)
plt.scatter(d.Overtime_Hours,d.Left_Company,marker='+',color='black')
plt.show()
from sklearn.model_selection import train_test_split
X_tr,X_t,y_tr,y_t=train_test_split(d[["Overtime_Hours"]],d.Left_Company,train_size=0.8,random_state=42)
print(X_t)
from sklearn.linear_model import LogisticRegression
m=LogisticRegression()
m.fit(X_tr,y_tr)
print(X_t)
y_P=m.predict(X_t)
print("Intercept: ",m.intercept_)
print("Coefficient: ",m.coef_)
print("Prediction-Probability: ",m.predict_proba(X_t))
print(y_P)
print("Accurecy Score: ",m.score(X_t,y_t))
print(X_t)
import math
def predict_fun(Hours):
    z = m.coef_[0][0] * Hours + m.intercept_[0]
    return 1 / (1 + math.exp(-z))
Hours=90
print("Hours: 90",predict_fun(Hours))
Hours=87
print("Hours: 87",predict_fun(Hours))
Hours=40
print("Hours: 40",predict_fun(Hours))    
Hours=35
print("Hours: 35",predict_fun(Hours)) 