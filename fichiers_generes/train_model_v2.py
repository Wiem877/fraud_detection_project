"""
Entraînement v3 - méthodologie corrigée avec 3 ensembles distincts :
- TRAIN      : apprentissage du modèle
- VALIDATION : comparer v1 vs v2 (features) ET choisir le meilleur seuil
- TEST       : évaluation finale, UNE SEULE FOIS, à la toute fin

Le jeu de test n'est utilisé qu'une seule fois, tout à la fin, pour donner
un score honnête et non biaisé par nos propres choix de réglages.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve
)
import joblib

TRAIN_PATH = "transactions_train.csv"
TEST_PATH = "transactions_test.csv"
MODEL_OUTPUT_PATH = "fraud_model_final.joblib"
THRESHOLD_OUTPUT_PATH = "best_threshold.txt"


# ---------------------------------------------------------------------------
# 1. Feature engineering (identique à avant)
# ---------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].between(0, 4).astype(int)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df_indexed = df.set_index("timestamp")
    velocity = df_indexed.groupby("user_id")["transaction_id"].rolling("5min").count()
    df["velocity_count_5min"] = velocity.reset_index(level=0, drop=True).values

    df["prev_country"] = df.groupby("user_id")["country"].shift(1)
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["minutes_since_prev"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60
    df["impossible_travel_flag"] = (
        (df["country"] != df["prev_country"]) &
        (df["minutes_since_prev"] <= 20) &
        df["prev_country"].notna()
    ).astype(int)
    return df


full_train_df = add_features(pd.read_csv(TRAIN_PATH))
test_df = add_features(pd.read_csv(TEST_PATH))  # <-- on n'y touche plus avant l'étape 6

numeric_features_v1 = ["amount", "hour_of_day", "is_night", "ip_is_datacenter"]
numeric_features_v2 = numeric_features_v1 + ["velocity_count_5min", "impossible_travel_flag"]
categorical_features = ["country", "merchant_category"]


def build_pipeline(numeric_features):
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42
        )),
    ])


# ---------------------------------------------------------------------------
# 2. Découpage TRAIN / VALIDATION (le jeu de test reste intouché)
# ---------------------------------------------------------------------------
train_part, val_part = train_test_split(
    full_train_df, test_size=0.25, stratify=full_train_df["is_fraud"], random_state=42
)
y_train_part = train_part["is_fraud"]
y_val_part = val_part["is_fraud"]

print(f"Train : {len(train_part)} | Validation : {len(val_part)} | "
      f"Test (intouché pour l'instant) : {len(test_df)}")

# ---------------------------------------------------------------------------
# 3. Comparer v1 vs v2 SUR LA VALIDATION (pas sur le test !)
# ---------------------------------------------------------------------------
print("\n--- Comparaison v1 (sans features avancées) vs v2 (avec) sur VALIDATION ---")

results = {}
for name, num_feats in [("v1", numeric_features_v1), ("v2", numeric_features_v2)]:
    cols = num_feats + categorical_features
    pipe = build_pipeline(num_feats)
    pipe.fit(train_part[cols], y_train_part)
    y_val_pred = pipe.predict(val_part[cols])
    f1 = f1_score(y_val_part, y_val_pred)
    results[name] = f1
    print(f"{name}: F1 (validation) = {f1:.4f} | "
          f"Precision = {precision_score(y_val_part, y_val_pred):.4f} | "
          f"Recall = {recall_score(y_val_part, y_val_pred):.4f}")

best_version = max(results, key=results.get)
best_numeric_features = numeric_features_v2 if best_version == "v2" else numeric_features_v1
print(f"\n>>> Version retenue : {best_version} (meilleur F1 en validation)")

feature_columns = best_numeric_features + categorical_features

# ---------------------------------------------------------------------------
# 4. Choisir le meilleur SEUIL sur la validation (toujours pas le test)
# ---------------------------------------------------------------------------
final_pipe_for_threshold = build_pipeline(best_numeric_features)
final_pipe_for_threshold.fit(train_part[feature_columns], y_train_part)
y_val_proba = final_pipe_for_threshold.predict_proba(val_part[feature_columns])[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val_part, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"\nMeilleur seuil (choisi sur la validation) : {best_threshold:.3f}")

# ---------------------------------------------------------------------------
# 5. Ré-entraîner sur TRAIN + VALIDATION réunis (plus de données = mieux)
# ---------------------------------------------------------------------------
# Une fois la version ET le seuil choisis, on peut réutiliser toutes les
# données disponibles (train + validation) pour l'entraînement final.
final_model = build_pipeline(best_numeric_features)
final_model.fit(full_train_df[feature_columns], full_train_df["is_fraud"])

# ---------------------------------------------------------------------------
# 6. UNE SEULE évaluation finale, honnête, sur le vrai test
# ---------------------------------------------------------------------------
X_test = test_df[feature_columns]
y_test = test_df["is_fraud"]
y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print("\n" + "=" * 55)
print(f"RÉSULTAT FINAL SUR LE TEST (version={best_version}, seuil={best_threshold:.3f})")
print("=" * 55)
print(f"Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_test_pred):.4f}")

cm = confusion_matrix(y_test, y_test_pred)
print("\nMatrice de confusion (TEST) :")
print(f"                 Prédit: Normal   Prédit: Fraude")
print(f"Réel: Normal     {cm[0][0]:<16} {cm[0][1]}")
print(f"Réel: Fraude     {cm[1][0]:<16} {cm[1][1]}")

# ---------------------------------------------------------------------------
# 7. Sauvegarde
# ---------------------------------------------------------------------------
joblib.dump(final_model, MODEL_OUTPUT_PATH)
with open(THRESHOLD_OUTPUT_PATH, "w") as f:
    f.write(str(best_threshold))
print(f"\nModèle final sauvegardé : {MODEL_OUTPUT_PATH}")
print(f"Seuil sauvegardé : {THRESHOLD_OUTPUT_PATH}")