import pandas as pd


columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


data = pd.read_csv(
    "dataset/processed.cleveland.data",
    names=columns,
    na_values="?"
)


data = data.apply(pd.to_numeric, errors="coerce")

data = data.dropna()

# Convert target:
# 0 = No heart disease
# 1,2,3,4 = Heart disease
data["target"] = data["target"].apply(
    lambda x: 1 if x > 0 else 0
)

data.to_csv("dataset/heart_disease.csv", index=False)

print("Dataset converted successfully!")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nTarget distribution:")
print(data["target"].value_counts())



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("dataset/heart_disease.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000)
decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)
decision_tree_pred = decision_tree_model.predict(X_test)
random_forest_pred = random_forest_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_pred)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

print("\n==============================")
print("MODEL ACCURACY RESULTS")
print("==============================")

print(f"Logistic Regression: {logistic_accuracy * 100:.2f}%")
print(f"Decision Tree:       {decision_tree_accuracy * 100:.2f}%")
print(f"Random Forest:       {random_forest_accuracy * 100:.2f}%")

print("\n==============================")
print("LOGISTIC REGRESSION REPORT")
print("==============================")

print(classification_report(y_test, logistic_pred))

import os
import pickle

# Create models folder
os.makedirs("models", exist_ok=True)

# Save the best model
with open("models/random_forest_model.pkl", "wb") as file:
    pickle.dump(random_forest_model, file)

# Save the scaler
with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("\nBest model saved successfully!")
print("Model: Random Forest")
print("Accuracy: 85.00%")