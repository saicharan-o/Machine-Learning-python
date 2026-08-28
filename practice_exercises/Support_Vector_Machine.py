import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("Student_SVM.csv")
print(d.head())

plt.scatter(d["study_hours"],d["attendance"],color="orange",marker='s')
plt.xlabel("study_hours")
plt.ylabel("attendance")
plt.show()

plt.scatter(d["practice_tests"],d["screen_time"],color='purple',marker='+')
plt.xlabel("practice_tests")
plt.ylabel("screen_time")
plt.show()

d0=d[d.target==0]
print(d0)
d1=d[d.target==1]
print(d1)
d2=d[d.target==2]
print(d2)

plt.scatter(d0["study_hours"],d0["attendance"],color="g",marker='*')
plt.xlabel("study_hours")
plt.ylabel("attendance")
plt.show()

plt.scatter(d0["practice_tests"],d0["screen_time"],color='r',marker='o')
plt.xlabel("practice_tests")
plt.ylabel("screen_time")
plt.show()

plt.scatter(d1["study_hours"],d1["attendance"],color="b",marker='.')
plt.xlabel("study_hours")
plt.ylabel("attendance")
plt.show()

plt.scatter(d1["practice_tests"],d1["screen_time"],color='k',marker='^')
plt.xlabel("practice_tests")
plt.ylabel("screen_time")
plt.show()

plt.scatter(d2["study_hours"],d2["attendance"],color="c",marker='v')
plt.xlabel("study_hours")
plt.ylabel("attendance")
plt.show()

plt.scatter(d2["practice_tests"],d2["screen_time"],color='m',marker='D')
plt.xlabel("practice_tests")
plt.ylabel("screen_time")
plt.show()

x=d.drop("target",axis=1)
y=d.target
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X_tr,X_t,y_tr,y_t=train_test_split(x,y,test_size=0.3)
md=SVC()
md.fit(X_tr,y_tr)
print("Accuray Score: ",md.score(X_t,y_t))

md=SVC(C=6)
md.fit(X_tr,y_tr)
print("Regularization Accuracy score: ",md.score(X_t,y_t))

md=SVC(gamma=40)
md.fit(X_tr,y_tr)
print("Gamma Accuracy score: ",md.score(X_t,y_t))

md=SVC(kernel="linear")
md.fit(X_tr,y_tr)
print("kernel Accuracy score: ",md.score(X_t,y_t))