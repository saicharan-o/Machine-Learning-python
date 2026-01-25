import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("Random_Forest_file.csv")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

plt.scatter(df["Age"], df["Purchased"])
plt.xlabel("Age")
plt.ylabel("Purchased")
plt.title("Age vs Purchased-Figure-1")
plt.show()

plt.scatter(df["Annual_Income"], df["Spending_Score"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending-Figure-2")
plt.show()

plt.scatter(df["Age"], df["Monthly_Spend"], c=df["Purchased"])
plt.xlabel("Age")
plt.ylabel("Monthly Spend")
plt.title("Age vs Monthly Spend (Colored by Purchased)-Figure-3")
plt.show()

df["Purchased"].value_counts().plot(kind="bar")
plt.title("Purchased Count-Figure-4")
plt.show()

df.groupby("Purchased")["Monthly_Spend"].mean().plot(kind="bar")
plt.title("Avg Monthly Spend vs Purchased-Figure-5")
plt.show()

plt.hist(df["Age"], bins=10)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution-Figure-6")
plt.show()

df.boxplot(column="Monthly_Spend", by="Purchased")
plt.title("Monthly Spend by Purchased-Figure-7")
plt.show()

df_sorted = df.sort_values("Age")
plt.plot(df_sorted["Age"], df_sorted["Monthly_Spend"])
plt.xlabel("Age")
plt.ylabel("Monthly Spend")
plt.title("Age vs Monthly Spend Trend-Figure-8")
plt.show()

from pandas.plotting import scatter_matrix

scatter_matrix(
    df[["Age","Annual_Income","Spending_Score","Monthly_Spend"]],
    figsize=(10,10)
)
plt.suptitle("Relationship Between Age, Income, Spending Score and Monthly Spend-Figure-9", fontsize=16)
plt.show()

df["Purchased"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Purchased Distribution-Figure-10")
plt.show()

corr = df.corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix-Figure-11")
plt.show()

plt.scatter(df["Age"], df["Purchased"], label="Age")
plt.scatter(df["Spending_Score"], df["Purchased"], label="Spending")
plt.legend()
plt.title("Relationship of Age and Spending Score with Purchase Outcome-Figure-12")
plt.show()


X_tr, X_t, y_tr, y_t = train_test_split(
    X, y, test_size=0.2, random_state=42
)

md = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

md.fit(X_tr, y_tr)

y_p = md.predict(X_t)
print(y_p)
print("Accuracy:", accuracy_score(y_t, y_p))
print("\nConfusion Matrix:\n", confusion_matrix(y_t, y_p))
print("\nClassification Report:\n", classification_report(y_t, y_p))

m = pd.DataFrame({
    "Feature": X.columns,
    "Importance": md.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", m)

plt.figure(figsize=(10, 6))

plt.scatter(
    m["Importance"],
    m["Feature"],
    s=200
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (Scatter Plot) - Random Forest-Figure-13")
plt.grid(True)
plt.show()

plt.barh(m["Feature"], m["Importance"])
plt.title("Feature Importance-Figure-14")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(range(len(m)), m["Importance"])
plt.xticks(range(len(m)), m["Feature"], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance Scatter-Figure-15")
plt.grid(True)
plt.show()
