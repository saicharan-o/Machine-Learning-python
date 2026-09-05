import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.joblib")

FEATURE_NAMES = [
    "age", "gender", "systolic_bp", "diastolic_bp", "resting_heart_rate",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "fasting_glucose",
    "bmi", "smoking_status", "physical_activity_hours", "family_history"
]

def load_inference_artifacts():
    """Loads trained model and preprocessing scaler."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}. Please run `python run_pipeline.py` first.")
    if not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(f"Scaler artifact not found at {SCALER_FILE}.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

def predict_patient_risk(patient_dict: dict) -> dict:
    """
    Computes disease risk probability and generates clinical risk stratification.
    """
    model, scaler = load_inference_artifacts()

    # Format into DataFrame with correct order
    df_input = pd.DataFrame([patient_dict])[FEATURE_NAMES]
    scaled_input = scaler.transform(df_input)

    prob = model.predict_proba(scaled_input)[0][1]
    prediction = int(model.predict(scaled_input)[0])

    if prob < 0.25:
        risk_tier = "Low Risk"
        color = "Green"
        recommendation = "Maintain regular annual check-ups, balanced diet, and 150+ minutes of weekly aerobic exercise."
    elif prob < 0.55:
        risk_tier = "Moderate Risk"
        color = "Yellow"
        recommendation = "Schedule biometric monitoring every 6 months. Consider dietary sodium restriction and lipid profile review."
    elif prob < 0.80:
        risk_tier = "High Risk"
        color = "Orange"
        recommendation = "Physician consultation recommended within 14 days. Evaluate pharmacological therapy for BP/Lipid management."
    else:
        risk_tier = "Critical Risk"
        color = "Red"
        recommendation = "Urgent clinical diagnostic workup recommended. Comprehensive cardiovascular evaluation advised."

    # Identify primary risk contributors
    risk_factors = []
    if patient_dict.get("systolic_bp", 120) >= 140 or patient_dict.get("diastolic_bp", 80) >= 90:
        risk_factors.append(f"Stage 2 Hypertension ({patient_dict.get('systolic_bp')}/{patient_dict.get('diastolic_bp')} mmHg)")
    if patient_dict.get("ldl_cholesterol", 100) >= 160:
        risk_factors.append(f"Elevated LDL Cholesterol ({patient_dict.get('ldl_cholesterol')} mg/dL)")
    if patient_dict.get("fasting_glucose", 90) >= 126:
        risk_factors.append(f"Hyperglycemia / Diabetic Range Glucose ({patient_dict.get('fasting_glucose')} mg/dL)")
    if patient_dict.get("bmi", 24) >= 30:
        risk_factors.append(f"Class I/II Obesity (BMI: {patient_dict.get('bmi')})")
    if patient_dict.get("smoking_status", 0) == 2:
        risk_factors.append("Active Tobacco Smoker")
    if patient_dict.get("family_history", 0) == 1:
        risk_factors.append("Positive Family History of Cardiovascular Disease")

    return {
        "prediction": prediction,
        "disease_probability": round(float(prob), 4),
        "disease_probability_percent": f"{prob*100:.1f}%",
        "risk_tier": risk_tier,
        "tier_color": color,
        "primary_risk_factors": risk_factors,
        "clinical_recommendation": recommendation
    }

def batch_predict(input_csv_path: str, output_csv_path: str):
    """Generates batch predictions for a CSV of patient records."""
    model, scaler = load_inference_artifacts()
    df = pd.read_csv(input_csv_path)

    X_scaled = scaler.transform(df[FEATURE_NAMES])
    df["predicted_high_risk"] = model.predict(X_scaled)
    df["risk_probability"] = model.predict_proba(X_scaled)[:, 1].round(4)
    df["risk_tier"] = pd.cut(
        df["risk_probability"],
        bins=[-0.01, 0.25, 0.55, 0.80, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
    )

    df.to_csv(output_csv_path, index=False)
    print(f"[OK] Batch predictions saved to: {output_csv_path}")
    return df

if __name__ == "__main__":
    sample_patient = {
        "age": 58,
        "gender": 1,
        "systolic_bp": 148,
        "diastolic_bp": 94,
        "resting_heart_rate": 82,
        "cholesterol_total": 245,
        "hdl_cholesterol": 38,
        "ldl_cholesterol": 168,
        "fasting_glucose": 134,
        "bmi": 31.4,
        "smoking_status": 2,
        "physical_activity_hours": 1.0,
        "family_history": 1
    }
    print("\n--- Sample Patient Risk Assessment ---")
    result = predict_patient_risk(sample_patient)
    for k, v in result.items():
        print(f"{k}: {v}")
