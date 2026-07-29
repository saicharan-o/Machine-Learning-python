import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#Load & Prepare Data
df = pd.read_csv("salary.csv")
X = df.drop(columns=["salary_gt_100k"])
y = df[["salary_gt_100k"]]

#Encode Categorical Features
enc_company=LabelEncoder()
enc_job=LabelEncoder()
enc_degree=LabelEncoder()
enc_experience=LabelEncoder()

X["company_n"]=enc_company.fit_transform(X["company"])
X["job_n"]=enc_job.fit_transform(X["job"])
X["degree_n"]=enc_degree.fit_transform(X["degree"])
X["experience_n"]=enc_experience.fit_transform(X["experience"])

X_encoded=X.drop(columns=["company","job","degree","experience"])
print(X_encoded)
print(y)

# Train Model
model=DecisionTreeClassifier()
model.fit(X_encoded, y)
print(f"Training Accuracy: {model.score(X_encoded, y):.2%}")

# Helper Functions
def decode_salary(val: int) -> str:
    return ">100K" if val==1 else "<=100K"


def print_prediction(title: str,company: int,job: int,degree: int,exp: int) -> None:
    input_df=pd.DataFrame({
        "company_n":[company],"job_n":[job],
        "degree_n":[degree],"experience_n":[exp]
    })
    pred=model.predict(input_df)[0]

    print(f"\n{title}")
    print(f"Company:{enc_company.inverse_transform([company])[0].title()}")
    print(f"Role:{enc_job.inverse_transform([job])[0].title()}")
    print(f"Degree:{enc_degree.inverse_transform([degree])[0].title()}")
    print(f"Experience:{enc_experience.inverse_transform([exp])[0].title()}")
    print(f"Salary:{decode_salary(pred)}")
    print("-"*40)


# Run Predictions
print_prediction("Prediction 1",company=1,job=1,degree=0,exp=0)
print_prediction("Prediction 2",company=2,job=2,degree=2,exp=1)
print_prediction("Prediction 3",company=1,job=2,degree=0,exp=1)
print_prediction("Prediction 4",company=0,job=1,degree=2,exp=1)
print_prediction("Prediction 5",company=2,job=2,degree=1,exp=1)
print_prediction("Prediction 6",company=1,job=2,degree=0,exp=1)