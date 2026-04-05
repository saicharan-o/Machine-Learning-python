import pandas as pd
d=pd.read_csv("salary.csv")
x=d.drop(d[["salary_gt_100k"]],axis="columns")
y=d[["salary_gt_100k"]]
from sklearn.preprocessing import LabelEncoder
l_c=LabelEncoder()
l_j=LabelEncoder()
l_d=LabelEncoder()
l_e=LabelEncoder()
x["company_n"]=l_c.fit_transform(x["company"])
x["job_n"]=l_j.fit_transform(x["job"])
x["degree_n"]=l_d.fit_transform(x["degree"])
x["experience_n"]=l_e.fit_transform(x["experience"])
print(x.head())
x_n=x.drop(x[["company","job","degree","experience"]],axis="columns")
print(x_n)
print(y)
from sklearn.tree import DecisionTreeClassifier
md=DecisionTreeClassifier()
md.fit(x_n,y)
print(md.score(x_n,y))
def decode_salary(val):
    return ">100K" if val == 1 else "<=100K"

p2 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[1],"degree_n":[0],"experience_n":[0]}))
print("Prediction 2:", decode_salary(p2[0]))

p3 = md.predict(pd.DataFrame({"company_n":[2],"job_n":[2],"degree_n":[2],"experience_n":[1]}))
print("Prediction 3:", decode_salary(p3[0]))

p4 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}))
print("Prediction 4:", decode_salary(p4[0]))

p5 = md.predict(pd.DataFrame({"company_n":[0],"job_n":[1],"degree_n":[2],"experience_n":[1]}))
print("Prediction 5:", decode_salary(p5[0]))

p6 = md.predict(pd.DataFrame({"company_n":[2],"job_n":[2],"degree_n":[1],"experience_n":[1]}))
print("Prediction 6:", decode_salary(p6[0]))

p7 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}))
print("Prediction 7:", decode_salary(p7[0]))