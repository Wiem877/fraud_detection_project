"""
Interface graphique (Streamlit) pour visualiser les performances du
modèle de détection de fraude, et tester des transactions manuellement.

Lancement :
streamlit run dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Détection de fraude - Dashboard",
    layout="wide"
)

MODEL_PATH = "fraud_model_final.joblib"
THRESHOLD_PATH = "best_threshold.txt"
TEST_PATH = "transactions_test.csv"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour_of_day"] = df["timestamp"].dt.hour

    df["is_night"] = (
        df["hour_of_day"].between(0, 4)
    ).astype(int)

    df = df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    df_indexed = df.set_index("timestamp")

    velocity = (
        df_indexed
        .groupby("user_id")["transaction_id"]
        .rolling("5min")
        .count()
    )

    df["velocity_count_5min"] = (
        velocity
        .reset_index(level=0, drop=True)
        .values
    )

    df["prev_country"] = (
        df.groupby("user_id")["country"].shift(1)
    )

    df["prev_timestamp"] = (
        df.groupby("user_id")["timestamp"].shift(1)
    )

    df["minutes_since_prev"] = (
        df["timestamp"] - df["prev_timestamp"]
    ).dt.total_seconds() / 60

    df["impossible_travel_flag"] = (
        (df["country"] != df["prev_country"]) &
        (df["minutes_since_prev"] <= 20) &
        df["prev_country"].notna()
    ).astype(int)

    return df


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def load_model(model_timestamp):
    return joblib.load(MODEL_PATH)


def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        return float(f.read().strip())


@st.cache_data
def load_test_data():
    return add_features(
        pd.read_csv(TEST_PATH)
    )


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "amount",
    "hour_of_day",
    "is_night",
    "ip_is_datacenter",
    "velocity_count_5min",
    "impossible_travel_flag",
    "country",
    "merchant_category"
]


# ============================================================
# CHARGEMENT
# ============================================================

model_timestamp = os.path.getmtime(MODEL_PATH)

model = load_model(model_timestamp)

best_threshold = load_threshold()

test_df = load_test_data()

X_test = test_df[feature_columns]

y_test = test_df["is_fraud"]

y_proba = model.predict_proba(X_test)[:, 1]


# ============================================================
# TITRE
# ============================================================

st.title("🔍 Dashboard - Détection de fraude en temps réel")

st.info(
    f"🎯 **Seuil optimisé utilisé par le modèle final : "
    f"{best_threshold:.3f}**"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Réglages")

st.sidebar.success(
    f"Seuil optimisé : **{best_threshold:.3f}**"
)

threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=float(best_threshold),
    step=0.01
)

if abs(threshold - best_threshold) < 0.005:
    st.sidebar.caption(
        "✅ Seuil officiel optimisé sur la validation"
    )
else:
    st.sidebar.warning(
        "⚠️ Seuil de simulation différent du seuil optimisé"
    )


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = (
    y_proba >= threshold
).astype(int)


# ============================================================
# METRIQUES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy_score(y_test, y_pred):.1%}"
)

col2.metric(
    "Precision",
    f"{precision_score(y_test, y_pred, zero_division=0):.1%}"
)

col3.metric(
    "Recall",
    f"{recall_score(y_test, y_pred, zero_division=0):.1%}"
)

col4.metric(
    "F1-score",
    f"{f1_score(y_test, y_pred, zero_division=0):.1%}"
)


# ============================================================
# MATRICE DE CONFUSION + COURBE
# ============================================================

col_left, col_right = st.columns(2)


# ------------------------------------------------------------
# Matrice de confusion
# ------------------------------------------------------------

with col_left:

    st.subheader("Matrice de confusion")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(4, 3.5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraude"],
        yticklabels=["Normal", "Fraude"],
        ax=ax
    )

    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")

    st.pyplot(fig)


# ------------------------------------------------------------
# Precision / Recall selon le seuil
# ------------------------------------------------------------

with col_right:

    st.subheader(
        "Courbe Precision / Recall selon le seuil"
    )

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_test,
            y_proba
        )
    )

    fig2, ax2 = plt.subplots(
        figsize=(4, 3.5)
    )

    ax2.plot(
        thresholds,
        precisions[:-1],
        label="Precision"
    )

    ax2.plot(
        thresholds,
        recalls[:-1],
        label="Recall"
    )

    # Seuil optimisé
    ax2.axvline(
        best_threshold,
        linestyle="--",
        label=f"Seuil optimisé ({best_threshold:.3f})"
    )

    # Seuil actuellement sélectionné
    if abs(threshold - best_threshold) > 0.005:

        ax2.axvline(
            threshold,
            linestyle=":",
            label=f"Seuil testé ({threshold:.2f})"
        )

    ax2.set_xlabel("Seuil")

    ax2.set_ylabel("Score")

    ax2.legend()

    st.pyplot(fig2)


# ============================================================
# IMPORTANCE DES FEATURES
# ============================================================

st.subheader(
    "Importance des features dans la décision du modèle"
)

classifier = model.named_steps["classifier"]

preprocessor = model.named_steps["preprocessor"]

cat_names = (
    preprocessor
    .named_transformers_["cat"]
    .get_feature_names_out(
        ["country", "merchant_category"]
    )
)

all_feature_names = [
    "amount",
    "hour_of_day",
    "is_night",
    "ip_is_datacenter",
    "velocity_count_5min",
    "impossible_travel_flag"
] + list(cat_names)

importances = pd.Series(
    classifier.feature_importances_,
    index=all_feature_names
)

importances = (
    importances
    .sort_values(ascending=False)
    .head(10)
)

fig3, ax3 = plt.subplots(
    figsize=(8, 4)
)

importances.plot(
    kind="barh",
    ax=ax3
)

ax3.invert_yaxis()

st.pyplot(fig3)


# ============================================================
# TEST MANUEL
# ============================================================

st.subheader(
    "Tester une transaction manuellement"
)

c1, c2, c3 = st.columns(3)

amount = c1.number_input(
    "Montant (TND)",
    value=200.0
)

hour = c2.slider(
    "Heure",
    0,
    23,
    14
)

velocity = c3.number_input(
    "Nb transactions récentes (5 min)",
    value=1,
    min_value=1
)

country = c1.text_input(
    "Pays",
    "TN"
)

category = c2.text_input(
    "Catégorie marchand",
    "grocery"
)

is_dc = c3.checkbox(
    "IP datacenter suspecte"
)

impossible = c1.checkbox(
    "Voyage impossible détecté"
)


if st.button("Prédire"):

    sample = pd.DataFrame([{

        "amount": amount,

        "hour_of_day": hour,

        "is_night": (
            1 if hour < 5 else 0
        ),

        "ip_is_datacenter": int(is_dc),

        "velocity_count_5min": velocity,

        "impossible_travel_flag": int(impossible),

        "country": country,

        "merchant_category": category

    }])

    proba = model.predict_proba(
        sample
    )[0, 1]

    st.write(
        f"**Probabilité de fraude : "
        f"{proba:.1%}**"
    )

    st.write(
        f"Seuil utilisé : "
        f"**{threshold:.3f}**"
    )

    if proba >= threshold:

        st.error(
            "🚨 Transaction classée FRAUDE"
        )

    else:

        st.success(
            "✅ Transaction classée NORMALE"
        )