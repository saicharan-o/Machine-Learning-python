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

p2 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[1],"degree_n":[0],"experience_n":[0]}))
print_prediction("Prediction 1", 1,1,0,0, p2[0])

p3 = md.predict(pd.DataFrame({"company_n":[2],"job_n":[2],"degree_n":[2],"experience_n":[1]}))
print_prediction("Prediction 2", 2,2,2,1, p3[0])

p4 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}))
print_prediction("Prediction 3", 1,2,0,1, p4[0])

# Prediction 5
p5 = md.predict(pd.DataFrame({"company_n":[0],"job_n":[1],"degree_n":[2],"experience_n":[1]}))
print_prediction("Prediction 5", 0,1,2,1, p5[0])

# Prediction 6
p6 = md.predict(pd.DataFrame({"company_n":[2],"job_n":[2],"degree_n":[1],"experience_n":[1]}))
print_prediction("Prediction 6", 2,2,1,1, p6[0])

# Prediction 7
p7 = md.predict(pd.DataFrame({"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}))
print_prediction("Prediction 7", 1,2,0,1, p7[0])    