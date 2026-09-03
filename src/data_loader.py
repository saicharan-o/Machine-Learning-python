import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "patient_health_records.csv")

def generate_synthetic_clinical_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic clinical dataset for cardiovascular and chronic disease prognosis.
    Features: age, gender, systolic_bp, diastolic_bp, cholesterol_total, hdl, ldl,
              fasting_glucose, bmi, smoking_status, physical_activity_hrs, family_history, resting_hr.
    """
    np.random.seed(random_state)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Demographic features
    age = np.random.normal(52, 14, n_samples).clip(20, 85).astype(int)
    gender = np.random.binomial(1, 0.52, n_samples) # 1: Male, 0: Female

    # Hemodynamics
    systolic_bp = (100 + 0.45 * age + np.random.normal(12, 15, n_samples) + gender * 4).clip(90, 200).astype(int)
    diastolic_bp = (0.65 * systolic_bp + np.random.normal(5, 7, n_samples)).clip(55, 125).astype(int)
    resting_hr = np.random.normal(72, 11, n_samples).clip(45, 115).astype(int)

    # Lipids & Metabolic markers
    cholesterol_total = np.random.normal(205, 38, n_samples).clip(120, 340).astype(int)
    hdl = np.random.normal(50 - 5 * gender, 12, n_samples).clip(25, 95).astype(int)
    ldl = (cholesterol_total - hdl - np.random.normal(30, 8, n_samples)).clip(50, 250).astype(int)
    fasting_glucose = (85 + 0.3 * (age - 30).clip(0) + np.random.exponential(15, n_samples)).clip(65, 260).astype(int)

    # Lifestyle & Body Metrics
    bmi = np.random.normal(27.8, 5.2, n_samples).clip(16.5, 48.0).round(1)
    smoking_status = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]) # 0: Never, 1: Former, 2: Active
    physical_activity_hrs = np.random.exponential(2.8, n_samples).clip(0, 18).round(1)
    family_history = np.random.binomial(1, 0.32, n_samples)

    # Latent Risk Calculation (Log-odds based on established Framingham / ASCVD risk functions)
    risk_score = (
        0.065 * (age - 45) +
        0.055 * (systolic_bp - 120) +
        0.035 * (ldl - 100) -
        0.045 * (hdl - 50) +
        0.040 * (fasting_glucose - 100) +
        0.095 * (bmi - 25) +
        1.10 * (smoking_status == 2) +
        0.45 * (smoking_status == 1) +
        1.25 * family_history -
        0.18 * physical_activity_hrs +
        0.50 * gender -
        3.2
    )

    # Probability via sigmoid
    prob_disease = 1 / (1 + np.exp(-risk_score))
    high_risk_disease = (np.random.rand(n_samples) < prob_disease).astype(int)

    df = pd.DataFrame({
        "patient_id": [f"PT-{10000 + i}" for i in range(n_samples)],
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "resting_heart_rate": resting_hr,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl,
        "ldl_cholesterol": ldl,
        "fasting_glucose": fasting_glucose,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "physical_activity_hours": physical_activity_hrs,
        "family_history": family_history,
        "high_risk_disease": high_risk_disease
    })

    df.to_csv(DATA_FILE, index=False)
    print(f"[OK] Clinical dataset generated and saved to: {DATA_FILE} ({len(df)} records)")
    return df

def load_and_preprocess_data(test_size: float = 0.2, random_state: int = 42):
    """
    Loads dataset, splits into train/test, performs feature scaling,
    and returns standardized feature sets and metadata.
    """
    if not os.path.exists(DATA_FILE):
        df = generate_synthetic_clinical_data(2000, random_state)
    else:
        df = pd.read_csv(DATA_FILE)

    feature_cols = [
        "age", "gender", "systolic_bp", "diastolic_bp", "resting_heart_rate",
        "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "fasting_glucose",
        "bmi", "smoking_status", "physical_activity_hours", "family_history"
    ]
    target_col = "high_risk_disease"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, feature_cols, df
    )

if __name__ == "__main__":
    generate_synthetic_clinical_data()
