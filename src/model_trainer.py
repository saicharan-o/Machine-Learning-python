import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def get_base_models():
    """Initializes standard baseline classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight="balanced", n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42),
        "Support Vector Machine": SVC(kernel="linear", probability=True, max_iter=1000, random_state=42)
    }

def train_and_tune_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trains all benchmark models, optimizes hyperparameters on the top ensemble,
    and returns trained estimators with evaluation probabilities.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_base_models()
    results = {}

    print("\n" + "="*60)
    print(" [HealthPredict AI] Multi-Model Clinical Benchmarking")
    print("="*60)

    for name, model in models.items():
        print(f"[*] Training {name:<25}...", end=" ", flush=True)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "train_acc": train_score,
            "test_acc": test_score
        }
        print(f"Done! (Train: {train_score*100:.1f}%, Test: {test_score*100:.1f}%)")

    # Hyperparameter Optimization on Champion Candidate (Gradient Boosting)
    print("\n[*] Performing Fine-Tuning on Gradient Boosting Ensemble...")
    param_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.06, 0.1],
        "max_depth": [3, 4]
    }
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_gb = grid_search.best_estimator_

    y_pred_gb = best_gb.predict(X_test_scaled)
    y_prob_gb = best_gb.predict_proba(X_test_scaled)[:, 1]

    results["Tuned Gradient Boosting (Champion)"] = {
        "model": best_gb,
        "y_pred": y_pred_gb,
        "y_prob": y_prob_gb,
        "train_acc": best_gb.score(X_train_scaled, y_train),
        "test_acc": best_gb.score(X_test_scaled, y_test),
        "best_params": grid_search.best_params_
    }
    print(f"[OK] Tuned Model Test Accuracy: {best_gb.score(X_test_scaled, y_test)*100:.2f}%")
    print(f"[OK] Optimal Hyperparameters: {grid_search.best_params_}")

    # Save champion model
    champion_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_gb, champion_path)
    print(f"[OK] Champion model saved to: {champion_path}")

    return results, best_gb
