import pickle
import numpy as np

with open("models/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

print("======================================")
print("   HEART DISEASE PREDICTION SYSTEM")
print("======================================")

print("\nEnter patient details:")

age = float(input("Age: "))
sex = float(input("Sex (1 = Male, 0 = Female): "))
cp = float(input("Chest Pain Type (1-4): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (1 = Yes, 0 = No): "))
restecg = float(input("Resting ECG Result (0-2): "))
thalach = float(input("Maximum Heart Rate: "))
exang = float(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
oldpeak = float(input("ST Depression: "))
slope = float(input("Slope (1-3): "))
ca = float(input("Number of Major Vessels (0-3): "))
thal = float(input("Thalassemia (3, 6, or 7): "))

patient_data = np.array([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

patient_scaled = scaler.transform(patient_data)

prediction = model.predict(patient_scaled)[0]

probability = model.predict_proba(patient_scaled)[0]

print("\n======================================")
print("             PREDICTION")
print("======================================")

if prediction == 1:
    print("Result: HEART DISEASE DETECTED")
else:
    print("Result: NO HEART DISEASE")

print(f"Probability of No Heart Disease: {probability[0] * 100:.2f}%")
print(f"Probability of Heart Disease: {probability[1] * 100:.2f}%")
print("======================================")