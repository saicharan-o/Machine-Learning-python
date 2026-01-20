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
from sklearn import tree
md=tree.DecisionTreeClassifier()
md.fit(x_n,y)
print(md.score(x_n,y))