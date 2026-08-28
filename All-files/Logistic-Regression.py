import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Load Data
df=pd.read_csv("Hours.csv")
print(df)

#Visualise Raw Data
plt.scatter(df.Overtime_Hours,df.Left_Company,marker='+',color='black')
plt.xlabel("Overtime Hours")
plt.ylabel("Left Company (1 = Yes)")
plt.title("Overtime Hours vs Employee Attrition")
plt.show()

#Train / Test Split
X_train,X_test,y_train,y_test=train_test_split(
    df[["Overtime_Hours"]],df.Left_Company,
    train_size=0.8,random_state=42
)

#Train Model
model=LogisticRegression()
model.fit(X_train,y_train)

#Evaluate
y_pred=model.predict(X_test)
print("Intercept            :",model.intercept_)
print("Coefficient          :",model.coef_)
print("Prediction Proba     :",model.predict_proba(X_test))
print("Predictions          :",y_pred)
print("Accuracy Score       :",model.score(X_test,y_test))

#Manual Sigmoid Verification
def predict_probability(hours: int) -> float:
    z=model.coef_[0][0]*hours+model.intercept_[0]
    return 1/(1+math.exp(-z))

for hrs in [90,87,40,35]:
    print(f"Hours: {hrs:>2} → P(leave) = {predict_probability(hrs):.4f}")
