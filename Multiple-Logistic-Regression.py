import pandas as pd
import matplotlib.pyplot as plt
d = pd.read_csv("Student.csv")
x=d[["Study_Hours","Attendance_Percent","Previous_Score","Backlogs"]]
y=d.Result
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_tr,X_t,y_tr,y_t=train_test_split(x,y,test_size=0.3)
m=LogisticRegression()
m.fit(X_tr,y_tr)
print(m.score(X_t,y_t))
print("Prediction 1: ",m.predict(pd.DataFrame({"Study_Hours":[18],"Attendance_Percent":[65],"Previous_Score":[70],"Backlogs":[7]})))
print("Prediction 2: ",m.predict(pd.DataFrame({"Study_Hours":[20],"Attendance_Percent":[50],"Previous_Score":[30],"Backlogs":[1]})))
print("Prediction 3: ",m.predict(pd.DataFrame({"Study_Hours":[5],"Attendance_Percent":[77],"Previous_Score":[20],"Backlogs":[5]})))
print("Prediction 4: ",m.predict(pd.DataFrame({"Study_Hours":[1],"Attendance_Percent":[45],"Previous_Score":[59],"Backlogs":[3]})))
print("Prediction 5: ",m.predict(pd.DataFrame({"Study_Hours":[10],"Attendance_Percent":[30],"Previous_Score":[67],"Backlogs":[0]})))

y_P=m.predict(X_t)
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_t,y_P)
print(c)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(c, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()