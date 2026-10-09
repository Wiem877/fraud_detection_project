"""
Entraînement V2 - Détection de fraude

Méthodologie :

1. transactions_train.csv
       ├── 80% TRAIN
       └── 20% VALIDATION

2. TRAIN
   -> apprentissage du Random Forest

3. VALIDATION
   -> évaluation
   -> choix du meilleur seuil
   -> aucune utilisation du TEST

4. TEST
   -> utilisé UNE SEULE FOIS après validation
   -> mesure finale du modèle

5. Sauvegarde :
   - fraud_model_final.joblib
   - best_threshold.txt
   - feature_columns.joblib
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_PATH = (
    r"C:\Users\dadiw\OneDrive\Desktop"
    r"\fraud_detection_project\fichiers_generes"
    r"\transactions_train.csv"
)

TEST_PATH = (
    r"C:\Users\dadiw\OneDrive\Desktop"
    r"\fraud_detection_project\fichiers_generes"
    r"\transactions_test.csv"
)

MODEL_PATH = "fraud_model_final.joblib"
THRESHOLD_PATH = "best_threshold.txt"
FEATURES_PATH = "feature_columns.joblib"


# ============================================================================
# 1. CHARGEMENT
# ============================================================================

print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

df_train_full = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"Train disponible : {len(df_train_full)}")
print(f"Test disponible  : {len(df_test)}")


# ============================================================================
# 2. VÉRIFICATION
# ============================================================================

print("\nRépartition des classes :")

print(
    df_train_full["is_fraud"]
    .value_counts()
)

print("\nPourcentages :")

print(
    df_train_full["is_fraud"]
    .value_counts(normalize=True)
    * 100
)


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def prepare_features(df, reference_df=None):

    data = df.copy()

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Heure nocturne
    data["is_night"] = (
        (data["hour"] >= 0)
        & (data["hour"] <= 5)
    ).astype(int)

    # ------------------------------------------------------------------------
    # CYCLICITÉ DE L'HEURE
    # ------------------------------------------------------------------------

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    # ------------------------------------------------------------------------
    # MONTANT
    # ------------------------------------------------------------------------

    data["amount_log"] = np.log1p(
        data["amount"]
    )

    # ------------------------------------------------------------------------
    # COUNTRY
    # ------------------------------------------------------------------------

    # Pays différent de la majorité historique
    # de l'utilisateur.
    if reference_df is not None:

        reference = reference_df.copy()

        reference["user_country"] = (
            reference
            .groupby("user_id")["country"]
            .transform(
                lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else x.iloc[0]
            )
        )

        usual_country = (
            reference
            .groupby("user_id")["user_country"]
            .first()
            .to_dict()
        )

        data["is_country_unusual"] = (
            data["country"]
            != data["user_id"].map(usual_country)
        ).astype(int)

    else:

        data["is_country_unusual"] = 0

    # ------------------------------------------------------------------------
    # FEATURES UTILISATEUR
    # ------------------------------------------------------------------------

    # Montant moyen historique de l'utilisateur.
    # La référence est construite uniquement à partir
    # des données d'entraînement.

    if reference_df is not None:

        user_avg_amount = (
            reference_df
            .groupby("user_id")["amount"]
            .mean()
            .to_dict()
        )

        data["user_avg_amount"] = (
            data["user_id"]
            .map(user_avg_amount)
        )

        # Pour les utilisateurs inconnus
        global_mean = reference_df["amount"].mean()

        data["user_avg_amount"] = (
            data["user_avg_amount"]
            .fillna(global_mean)
        )

    else:

        data["user_avg_amount"] = (
            data["amount"].mean()
        )

    # Rapport montant / moyenne utilisateur
    data["amount_ratio"] = (
        data["amount"]
        / data["user_avg_amount"].replace(0, 1)
    )

    # ------------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------------

    # Nombre de devices connus pour chaque utilisateur
    if reference_df is not None:

        user_devices = (
            reference_df
            .groupby("user_id")["device_id"]
            .apply(set)
            .to_dict()
        )

        data["is_new_device"] = data.apply(
            lambda row:
                int(
                    row["device_id"]
                    not in user_devices.get(
                        row["user_id"],
                        set()
                    )
                ),
            axis=1
        )

    else:

        data["is_new_device"] = 0

    # ------------------------------------------------------------------------
    # SUPPRESSION DES COLONNES NON UTILISÉES
    # ------------------------------------------------------------------------

    columns_to_drop = [
        "transaction_id",
        "user_id",
        "timestamp",
        "device_id",
        "is_fraud",
        "fraud_type",
        "currency"
    ]

    for col in columns_to_drop:

        if col in data.columns:
            data = data.drop(columns=col)

    # ------------------------------------------------------------------------
    # ENCODAGE DES VARIABLES CATÉGORIELLES
    # ------------------------------------------------------------------------

    categorical_columns = []

    if "country" in data.columns:
        categorical_columns.append("country")

    if "merchant_category" in data.columns:
        categorical_columns.append(
            "merchant_category"
        )

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        dtype=int
    )

    return data


# ============================================================================
# 4. SPLIT TRAIN / VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("SPLIT TRAIN / VALIDATION")
print("=" * 70)

# IMPORTANT :
# df_train_full est déjà chronologique.
#
# On garde donc l'ordre temporel.
#
# 80% -> TRAIN
# 20% -> VALIDATION

split_index = int(
    len(df_train_full) * 0.80
)

df_train = df_train_full.iloc[
    :split_index
].copy()

df_validation = df_train_full.iloc[
    split_index:
].copy()


print(
    f"TRAIN      : {len(df_train)}"
)

print(
    f"VALIDATION : {len(df_validation)}"
)

print(
    f"TEST       : {len(df_test)}"
)


# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# IMPORTANT :
# Les informations historiques utilisées pour créer
# certaines features viennent UNIQUEMENT du TRAIN.

X_train = prepare_features(
    df_train,
    reference_df=df_train
)

X_validation = prepare_features(
    df_validation,
    reference_df=df_train
)

X_test = prepare_features(
    df_test,
    reference_df=df_train
)

y_train = df_train["is_fraud"]

y_validation = df_validation["is_fraud"]

y_test = df_test["is_fraud"]


# ============================================================================
# 6. ALIGNEMENT DES FEATURES
# ============================================================================

# Le validation/test peuvent contenir un pays ou une catégorie
# absente du TRAIN.
#
# On force donc exactement les mêmes colonnes.

X_validation = X_validation.reindex(
    columns=X_train.columns,
    fill_value=0
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


print(
    f"\nNombre de features : {X_train.shape[1]}"
)


# ============================================================================
# 7. RANDOM FOREST
# ============================================================================

print("\n" + "=" * 70)
print("ENTRAÎNEMENT RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(

    # Nombre d'arbres
    n_estimators=300,

    # Contrôle de la profondeur
    max_depth=None,

    # Minimum d'échantillons pour diviser un noeud
    min_samples_split=4,

    # Minimum d'échantillons dans une feuille
    min_samples_leaf=2,

    # Sélection aléatoire des features
    max_features="sqrt",

    # Important pour le déséquilibre fraude / normal
    class_weight="balanced_subsample",

    # Reproductibilité
    random_state=42,

    # Utilisation des CPU
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Entraînement terminé.")


# ============================================================================
# 8. PRÉDICTION SUR VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("ÉVALUATION SUR VALIDATION")
print("=" * 70)

validation_probabilities = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================================
# 9. ROC-AUC VALIDATION
# ============================================================================

validation_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)

print(
    f"\nValidation ROC-AUC : "
    f"{validation_auc:.4f}"
)


# ============================================================================
# 10. RECHERCHE DU MEILLEUR SEUIL
# ============================================================================

print("\nRecherche du meilleur seuil...")

best_threshold = 0.50
best_f1 = 0

results_thresholds = []

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    validation_predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    results_thresholds.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print(
    f"\nMeilleur seuil : "
    f"{best_threshold:.2f}"
)

print(
    f"Meilleur F1 validation : "
    f"{best_f1:.4f}"
)


# ============================================================================
# 11. RAPPORT VALIDATION
# ============================================================================

validation_predictions = (
    validation_probabilities
    >= best_threshold
).astype(int)


print("\nClassification Report - VALIDATION")

print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=[
            "Normal",
            "Fraude"
        ],
        zero_division=0
    )
)


print("Confusion Matrix - VALIDATION")

print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


# ============================================================================
# 12. DÉCISION : LE MODÈLE EST-IL ACCEPTABLE ?
# ============================================================================

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


print("\n" + "=" * 70)
print("DIAGNOSTIC VALIDATION")
print("=" * 70)

print(
    f"Precision : {validation_precision:.4f}"
)

print(
    f"Recall    : {validation_recall:.4f}"
)

print(
    f"F1-score  : {validation_f1:.4f}"
)


# Seuil indicatif pour éviter de tester un modèle manifestement faible.
#
# Ce n'est PAS une vérité scientifique universelle.
# C'est simplement une alerte pour ton projet.

MIN_F1 = 0.50


if validation_f1 < MIN_F1:

    print(
        "\n⚠️ MODÈLE NON SATISFAISANT."
    )

    print(
        "Le modèle n'est pas envoyé au TEST."
    )

    print(
        "Il faut améliorer les données/features/"
        "paramètres puis réentraîner."
    )

    raise SystemExit


print(
    "\n✓ Validation suffisamment bonne."
)

print(
    "Le modèle peut maintenant être évalué sur le TEST."
)


# ============================================================================
# 13. TEST FINAL
# ============================================================================

print("\n" + "=" * 70)
print("ÉVALUATION FINALE SUR TEST")
print("=" * 70)

test_probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


print(
    f"\nTEST ROC-AUC : "
    f"{test_auc:.4f}"
)

print(
    f"TEST Precision : "
    f"{test_precision:.4f}"
)

print(
    f"TEST Recall : "
    f"{test_recall:.4f}"
)

print(
    f"TEST F1-score : "
    f"{test_f1:.4f}"
)


print("\nClassification Report - TEST")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Normal",
            "Fraude"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix - TEST")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)


# ============================================================================
# 14. IMPORTANCE DES FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("IMPORTANCE DES FEATURES")
print("=" * 70)

feature_importance = pd.DataFrame({

    "feature":
        X_train.columns,

    "importance":
        model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ============================================================================
# 15. SAUVEGARDE
# ============================================================================

print("\n" + "=" * 70)
print("SAUVEGARDE DU MODÈLE")
print("=" * 70)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    list(X_train.columns),
    FEATURES_PATH
)

with open(
    THRESHOLD_PATH,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


print(
    f"✓ Modèle : {MODEL_PATH}"
)

print(
    f"✓ Seuil  : {THRESHOLD_PATH}"
)

print(
    f"✓ Features : {FEATURES_PATH}"
)


# ============================================================================
# 16. RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ V2")
print("=" * 70)

print(
    f"Train      : {len(df_train)}"
)

print(
    f"Validation : {len(df_validation)}"
)

print(
    f"Test       : {len(df_test)}"
)

print(
    f"\nSeuil choisi sur validation : "
    f"{best_threshold:.2f}"
)

print(
    f"\nValidation F1 : "
    f"{validation_f1:.4f}"
)

print(
    f"Test F1       : "
    f"{test_f1:.4f}"
)

print(
    f"\nValidation Recall : "
    f"{validation_recall:.4f}"
)

print(
    f"Test Recall       : "
    f"{test_recall:.4f}"
)

print(
    "\n✓ Entraînement V2 terminé."
)