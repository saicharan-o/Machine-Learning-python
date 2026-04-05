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

def decode_input(company, job, degree, exp):
    return {
        "company": l_c.inverse_transform([company])[0],
        "job": l_j.inverse_transform([job])[0],
        "degree": l_d.inverse_transform([degree])[0],
        "experience": l_e.inverse_transform([exp])[0]
    }
def print_prediction(title, company, job, degree, exp, pred):
    info = decode_input(company, job, degree, exp)
    salary = decode_salary(pred)

    print(f"{title}")
    print(f"Company     : {info['company'].title()}")
    print(f"Role        : {info['job'].title()}")
    print(f"Degree      : {info['degree'].title()}")
    print(f"Experience  : {info['experience'].title()}")
    print(f"Prediction  : {salary}")
    print("-" * 40)

    