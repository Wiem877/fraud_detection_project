"""
REAL-TIME FRAUD DETECTION DASHBOARD

Kafka
  ↓
Spark Structured Streaming
  ↓
Random Forest
  ↓
fichiers_generes/stream_results/batch_*.csv
  ↓
Streamlit
"""

import os
import glob
import time
import re

import pandas as pd
import streamlit as st


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "stream_results"
)


# ============================================================================
# PAGE
# ============================================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🚨",
    layout="wide"
)


# ============================================================================
# STYLE
# ============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# TITRE
# ============================================================================

st.markdown(
    '<div class="main-title">🚨 REAL-TIME FRAUD DETECTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Kafka → Spark Structured Streaming → Random Forest → Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Contrôle")

auto_refresh = st.sidebar.checkbox(
    "🔄 Actualisation automatique",
    value=True
)

refresh_seconds = st.sidebar.slider(
    "Intervalle de rafraîchissement",
    min_value=1,
    max_value=10,
    value=2
)

show_only_fraud = st.sidebar.checkbox(
    "🚨 Afficher uniquement les fraudes",
    value=False
)


# ============================================================================
# CHARGER TOUS LES BATCHES
# ============================================================================

def get_batch_files():

    files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "batch_*.csv"
        )
    )

    def batch_number(path):

        filename = os.path.basename(path)

        match = re.search(
            r"batch_(\d+)\.csv",
            filename
        )

        if match:
            return int(match.group(1))

        return -1

    files.sort(key=batch_number)

    return files


def load_results():

    files = get_batch_files()

    if not files:
        return pd.DataFrame(), [], -1

    dataframes = []

    for file in files:

        try:

            df_batch = pd.read_csv(file)

            # Ajouter le numéro du batch
            filename = os.path.basename(file)

            match = re.search(
                r"batch_(\d+)\.csv",
                filename
            )

            if match:
                batch_id = int(match.group(1))
            else:
                batch_id = -1

            df_batch["batch_id"] = batch_id

            dataframes.append(df_batch)

        except Exception as e:

            print(
                f"Erreur lecture {file}: {e}"
            )

    if not dataframes:

        return pd.DataFrame(), files, -1

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    last_batch = df["batch_id"].max()

    return df, files, last_batch


# ============================================================================
# CHARGEMENT
# ============================================================================

df, batch_files, last_batch = load_results()


# ============================================================================
# AUCUNE DONNEE
# ============================================================================

if df.empty:

    st.warning(
        "⏳ En attente des résultats de Spark..."
    )

    st.write(
        f"📂 Dossier surveillé : `{RESULTS_DIR}`"
    )

    st.write(
        "Le dashboard attend les fichiers `batch_*.csv`."
    )

    if auto_refresh:

        time.sleep(refresh_seconds)

        st.rerun()

    st.stop()


# ============================================================================
# NETTOYAGE
# ============================================================================

if "prediction" in df.columns:

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    ).fillna(0).astype(int)


if "fraud_probability" in df.columns:

    df["fraud_probability"] = pd.to_numeric(
        df["fraud_probability"],
        errors="coerce"
    ).fillna(0)


# ============================================================================
# DETECTION DU DERNIER BATCH
# ============================================================================

last_batch_df = df[
    df["batch_id"] == last_batch
].copy()


# ============================================================================
# INDICATEUR LIVE
# ============================================================================

st.success(
    f"🟢 PIPELINE ACTIF — Dernier batch reçu : **Batch {last_batch}**"
)

col_a, col_b, col_c = st.columns(3)

with col_a:

    st.metric(
        "📦 Batches reçus",
        len(batch_files)
    )

with col_b:

    st.metric(
        "⚡ Transactions du dernier batch",
        len(last_batch_df)
    )

with col_c:

    st.metric(
        "🚨 Fraudes du dernier batch",
        int(
            (last_batch_df["prediction"] == 1).sum()
        )
    )


# ============================================================================
# DERNIER BATCH
# ============================================================================

st.divider()

st.subheader(
    f"📦 Dernier batch analysé — Batch {last_batch}"
)

st.caption(
    f"{len(last_batch_df)} transaction(s) analysée(s) par Spark dans ce batch."
)


# ============================================================================
# STATISTIQUES DU BATCH
# ============================================================================

batch_fraud_count = int(
    (last_batch_df["prediction"] == 1).sum()
)

batch_normal_count = int(
    (last_batch_df["prediction"] == 0).sum()
)


b1, b2, b3 = st.columns(3)

with b1:

    st.metric(
        "Transactions",
        len(last_batch_df)
    )

with b2:

    st.metric(
        "🚨 Fraudes",
        batch_fraud_count
    )

with b3:

    st.metric(
        "✅ Normales",
        batch_normal_count
    )


# ============================================================================
# TRANSACTIONS DU DERNIER BATCH
# ============================================================================

display_df = last_batch_df.copy()

if show_only_fraud:

    display_df = display_df[
        display_df["prediction"] == 1
    ]


display_columns = [
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "country",
    "merchant_category",
    "velocity_count_5min",
    "impossible_travel_flag",
    "fraud_probability",
    "prediction_label"
]


display_columns = [
    c
    for c in display_columns
    if c in display_df.columns
]


display_df = display_df[
    display_columns
].iloc[::-1]


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================================
# DERNIERE TRANSACTION DU BATCH
# ============================================================================

st.divider()

st.subheader(
    "⚡ Dernière transaction analysée"
)

last = last_batch_df.iloc[-1]


if last["prediction"] == 1:

    st.error(
        f"""
🚨 **FRAUDE DÉTECTÉE**

**Transaction :** `{last["transaction_id"]}`

**Utilisateur :** `{last["user_id"]}`

**Montant :** `{last["amount"]:.2f}`

**Pays :** `{last["country"]}`

**Probabilité de fraude :** `{last["fraud_probability"]:.2%}`
"""
    )

else:

    st.success(
        f"""
✅ **TRANSACTION NORMALE**

**Transaction :** `{last["transaction_id"]}`

**Utilisateur :** `{last["user_id"]}`

**Montant :** `{last["amount"]:.2f}`

**Pays :** `{last["country"]}`

**Probabilité de fraude :** `{last["fraud_probability"]:.2%}`
"""
    )


# ============================================================================
# INDICATEURS CRITIQUES
# ============================================================================

st.divider()

st.subheader(
    "🔎 Indicateurs comportementaux"
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "💰 Montant",
        f'{last["amount"]:.2f}'
    )


with c2:

    velocity = last.get(
        "velocity_count_5min",
        0
    )

    st.metric(
        "⚡ Transactions / 5 min",
        int(velocity)
    )


with c3:

    impossible = last.get(
        "impossible_travel_flag",
        0
    )

    if int(impossible) == 1:

        st.error(
            "🌍 Voyage impossible"
        )

    else:

        st.success(
            "🌍 Voyage normal"
        )


with c4:

    ip_dc = last.get(
        "ip_is_datacenter",
        0
    )

    if int(ip_dc) == 1:

        st.warning(
            "🖥️ IP Datacenter"
        )

    else:

        st.success(
            "🌐 IP classique"
        )


# ============================================================================
# PROBABILITE
# ============================================================================

st.subheader(
    "🎯 Probabilité de fraude"
)

probability = float(
    last["fraud_probability"]
)

st.progress(
    min(
        max(probability, 0),
        1
    )
)

st.write(
    f"Probabilité : **{probability:.2%}**"
)


# ============================================================================
# HISTORIQUE COMPLET
# ============================================================================

st.divider()

st.subheader(
    "📚 Historique des transactions"
)

history_df = df.copy()

if show_only_fraud:

    history_df = history_df[
        history_df["prediction"] == 1
    ]


history_columns = [
    "batch_id",
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "country",
    "merchant_category",
    "velocity_count_5min",
    "impossible_travel_flag",
    "fraud_probability",
    "prediction_label"
]


history_columns = [
    c
    for c in history_columns
    if c in history_df.columns
]


history_df = history_df[
    history_columns
].tail(100).iloc[::-1]


st.dataframe(
    history_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================================
# FRAUDES RECENTES
# ============================================================================

st.divider()

st.subheader(
    "🚨 Fraudes détectées récemment"
)

frauds = df[
    df["prediction"] == 1
].copy()


if frauds.empty:

    st.info(
        "Aucune fraude détectée pour le moment."
    )

else:

    frauds = frauds.tail(10).iloc[::-1]

    fraud_columns = [
        "batch_id",
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "country",
        "velocity_count_5min",
        "impossible_travel_flag",
        "fraud_probability",
        "prediction_label"
    ]

    fraud_columns = [
        c
        for c in fraud_columns
        if c in frauds.columns
    ]

    st.dataframe(
        frauds[fraud_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# EVOLUTION
# ============================================================================

st.divider()

st.subheader(
    "📈 Évolution des prédictions"
)

chart_df = (
    df
    .groupby("batch_id")
    .agg(
        Transactions=("prediction", "count"),
        Fraudes=("prediction", "sum")
    )
    .reset_index()
)


st.line_chart(
    chart_df.set_index("batch_id")
)


# ============================================================================
# PIPELINE
# ============================================================================

st.divider()

st.subheader(
    "🔗 Pipeline temps réel"
)

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.success("🟢 Kafka\n\ntransactions")

with p2:
    st.success("🟢 Spark\n\nStreaming")

with p3:
    st.success("🟢 Random Forest\n\nPrediction")

with p4:
    st.success(
        f"🟢 Dashboard\n\nBatch {last_batch}"
    )


# ============================================================================
# AUTO REFRESH
# ============================================================================

if auto_refresh:

    time.sleep(
        refresh_seconds
    )

    st.rerun()