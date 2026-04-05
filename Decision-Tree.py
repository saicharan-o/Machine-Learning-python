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

# Prediction 2
data = {"company_n":[1],"job_n":[1],"degree_n":[0],"experience_n":[0]}
p2 = md.predict(pd.DataFrame(data))
info = decode_input(1,1,0,0)

print("Prediction 2:")
print("Input:", info)
print("Salary:", decode_salary(p2[0]))
print()


# Prediction 3
data = {"company_n":[2],"job_n":[2],"degree_n":[2],"experience_n":[1]}
p3 = md.predict(pd.DataFrame(data))
info = decode_input(2,2,2,1)

print("Prediction 3:")
print("Input:", info)
print("Salary:", decode_salary(p3[0]))
print()


# Prediction 4
data = {"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}
p4 = md.predict(pd.DataFrame(data))
info = decode_input(1,2,0,1)

print("Prediction 4:")
print("Input:", info)
print("Salary:", decode_salary(p4[0]))
print()


# Prediction 5
data = {"company_n":[0],"job_n":[1],"degree_n":[2],"experience_n":[1]}
p5 = md.predict(pd.DataFrame(data))
info = decode_input(0,1,2,1)

print("Prediction 5:")
print("Input:", info)
print("Salary:", decode_salary(p5[0]))
print()


# Prediction 6
data = {"company_n":[2],"job_n":[2],"degree_n":[1],"experience_n":[1]}
p6 = md.predict(pd.DataFrame(data))
info = decode_input(2,2,1,1)

print("Prediction 6:")
print("Input:", info)
print("Salary:", decode_salary(p6[0]))
print()


# Prediction 7
data = {"company_n":[1],"job_n":[2],"degree_n":[0],"experience_n":[1]}
p7 = md.predict(pd.DataFrame(data))
info = decode_input(1,2,0,1)

print("Prediction 7:")
print("Input:", info)
print("Salary:", decode_salary(p7[0]))
print()