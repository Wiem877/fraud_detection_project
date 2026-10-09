"""
Spark Structured Streaming + Random Forest
Kafka -> Spark -> Feature Engineering -> Random Forest -> Dashboard

Les features du modèle sont exactement celles utilisées
dans train_model_v2.py.

velocity_count_5min et impossible_travel_flag sont calculées
comme indicateurs supplémentaires pour le dashboard.
"""

import os
import json
import glob
import shutil
from collections import defaultdict, deque
PYTHON_EXE = r"C:\ProgramData\anaconda3\envs\fraud_spark\python.exe"

os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StringType,
    DoubleType,
    IntegerType
)
from pyspark.sql.functions import (
    from_json,
    col
)


# ============================================================================
# CONFIGURATION
# ============================================================================

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

MODEL_PATH = "fichiers_generes/fraud_model_final.joblib"
FEATURES_PATH = "fichiers_generes/feature_columns.joblib"
THRESHOLD_PATH = "fichiers_generes/best_threshold.txt"

TRAIN_PATH = (
    r"C:\Users\dadiw\OneDrive\Desktop"
    r"\fraud_detection_project\fichiers_generes"
    r"\transactions_train.csv"
)

OUTPUT_DIR = "fichiers_generes/stream_results"

CHECKPOINT_DIR = r"C:\spark_checkpoints\fraud_prediction_v2"


# ============================================================================
# HADOOP WINDOWS
# ============================================================================

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = (
    os.environ["HADOOP_HOME"]
    + r"\bin;"
    + os.environ["PATH"]
)
os.environ["HADOOP_USER_NAME"] = "dadiw"


# ============================================================================
# NETTOYAGE DES ANCIENS RESULTATS
# ============================================================================

print("Préparation du dossier des résultats...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(CHECKPOINT_DIR):
    print("Suppression de l'ancien checkpoint...")
    shutil.rmtree(CHECKPOINT_DIR)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# CHARGEMENT DU MODELE
# ============================================================================

print("=" * 70)
print("CHARGEMENT DU RANDOM FOREST")
print("=" * 70)

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURES_PATH)

with open(THRESHOLD_PATH, "r") as f:
    best_threshold = float(f.read().strip())

print(f"Modèle chargé : {MODEL_PATH}")
print(f"Nombre de features : {len(feature_columns)}")
print(f"Seuil : {best_threshold:.2f}")


# ============================================================================
# CHARGEMENT DU TRAIN POUR LES FEATURES HISTORIQUES
# ============================================================================

print("\nChargement des données TRAIN...")

df_reference = pd.read_csv(TRAIN_PATH)

df_reference["timestamp"] = pd.to_datetime(
    df_reference["timestamp"]
)


# ============================================================================
# INFORMATIONS HISTORIQUES PAR UTILISATEUR
# ============================================================================

print("Construction des références utilisateurs...")


# Pays habituel
reference_user_country = (
    df_reference
    .groupby("user_id")["country"]
    .agg(
        lambda x: x.mode().iloc[0]
        if not x.mode().empty
        else x.iloc[0]
    )
    .to_dict()
)


# Montant moyen
reference_user_avg_amount = (
    df_reference
    .groupby("user_id")["amount"]
    .mean()
    .to_dict()
)


# Devices connus
reference_user_devices = (
    df_reference
    .groupby("user_id")["device_id"]
    .apply(set)
    .to_dict()
)


GLOBAL_MEAN_AMOUNT = df_reference["amount"].mean()


print("Références utilisateurs prêtes.")


# ============================================================================
# ETAT POUR VELOCITE ET VOYAGE IMPOSSIBLE
# ============================================================================

# user_id -> liste des timestamps récents
user_transactions_history = defaultdict(deque)

# user_id -> dernières transactions
user_last_transactions = defaultdict(deque)


# ============================================================================
# FONCTION DE FEATURE ENGINEERING
# ============================================================================

def prepare_features_streaming(df):
    """
    Reproduit le feature engineering de train_model_v2.py.
    """

    data = df.copy()

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["is_night"] = (
        (data["hour"] >= 0)
        & (data["hour"] <= 5)
    ).astype(int)


    # ------------------------------------------------------------------------
    # CYCLICITE
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
    # PAYS HABITUEL
    # ------------------------------------------------------------------------

    data["is_country_unusual"] = (
        data["country"]
        != data["user_id"].map(
            reference_user_country
        )
    ).astype(int)


    # ------------------------------------------------------------------------
    # MOYENNE UTILISATEUR
    # ------------------------------------------------------------------------

    data["user_avg_amount"] = (
        data["user_id"]
        .map(reference_user_avg_amount)
        .fillna(GLOBAL_MEAN_AMOUNT)
    )


    # ------------------------------------------------------------------------
    # RATIO MONTANT
    # ------------------------------------------------------------------------

    data["amount_ratio"] = (
        data["amount"]
        / data["user_avg_amount"].replace(0, 1)
    )


    # ------------------------------------------------------------------------
    # NOUVEAU DEVICE
    # ------------------------------------------------------------------------

    data["is_new_device"] = data.apply(
        lambda row:
            int(
                row["device_id"]
                not in reference_user_devices.get(
                    row["user_id"],
                    set()
                )
            ),
        axis=1
    )


    # ------------------------------------------------------------------------
    # COLONNES INUTILES
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

    for c in columns_to_drop:

        if c in data.columns:
            data = data.drop(columns=c)


    # ------------------------------------------------------------------------
    # ENCODAGE CATEGORIEL
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


    # ------------------------------------------------------------------------
    # EXACTEMENT LES FEATURES DU TRAIN
    # ------------------------------------------------------------------------

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


# ============================================================================
# FONCTION VELOCITE + VOYAGE IMPOSSIBLE
# ============================================================================

def calculate_streaming_indicators(df):
    """
    Calcule :

    - velocity_count_5min
    - impossible_travel_flag

    Ces deux indicateurs ne sont PAS envoyés au Random Forest.
    Ils sont uniquement affichés dans le dashboard.
    """

    velocity_values = []
    impossible_values = []


    for _, row in df.iterrows():

        user_id = row["user_id"]

        timestamp = pd.to_datetime(
            row["timestamp"],
            errors="coerce"
        )

        country = row["country"]


        # ================================================================
        # VELOCITE
        # ================================================================

        history = user_transactions_history[user_id]

        # supprimer les transactions de plus de 5 minutes
        while (
            history
            and timestamp - history[0]
            > pd.Timedelta(minutes=5)
        ):
            history.popleft()

        # ajouter transaction actuelle
        history.append(timestamp)

        velocity_count = len(history)

        velocity_values.append(
            velocity_count
        )


        # ================================================================
        # VOYAGE IMPOSSIBLE
        # ================================================================

        impossible = 0

        previous_transactions = (
            user_last_transactions[user_id]
        )

        for previous in previous_transactions:

            previous_time = previous["timestamp"]
            previous_country = previous["country"]

            time_difference = (
                timestamp - previous_time
            )

            if (
                time_difference >= pd.Timedelta(0)
                and
                time_difference <= pd.Timedelta(minutes=20)
                and
                previous_country != country
            ):

                impossible = 1
                break


        impossible_values.append(
            impossible
        )


        # conserver la transaction
        previous_transactions.append(
            {
                "timestamp": timestamp,
                "country": country
            }
        )


        # seulement garder les 20 dernières
        while len(previous_transactions) > 20:
            previous_transactions.popleft()


    df["velocity_count_5min"] = velocity_values

    df["impossible_travel_flag"] = impossible_values

    return df


# ============================================================================
# TRAITEMENT D'UN MICRO-BATCH
# ============================================================================

def process_batch(batch_df, batch_id):

    if batch_df.rdd.isEmpty():
        return


    print("\n")
    print("=" * 70)
    print(f"MICRO-BATCH {batch_id}")
    print("=" * 70)


    # Spark -> Pandas
    pdf = batch_df.toPandas()


    # ------------------------------------------------------------------------
    # INDICATEURS STREAMING
    # ------------------------------------------------------------------------

    pdf = calculate_streaming_indicators(
        pdf
    )


    # ------------------------------------------------------------------------
    # FEATURES DU MODELE
    # ------------------------------------------------------------------------

    X = prepare_features_streaming(
        pdf
    )


    # ------------------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------------------

    probabilities = (
        model
        .predict_proba(X)[:, 1]
    )


    predictions = (
        probabilities
        >= best_threshold
    ).astype(int)


    pdf["fraud_probability"] = (
        probabilities
    )

    pdf["prediction"] = predictions

    pdf["prediction_label"] = np.where(
        predictions == 1,
        "FRAUD",
        "NORMAL"
    )


    # ------------------------------------------------------------------------
    # AFFICHAGE CONSOLE
    # ------------------------------------------------------------------------

    for _, row in pdf.iterrows():

        print(
            f"TX={row['transaction_id']} | "
            f"USER={row['user_id']} | "
            f"AMOUNT={row['amount']:.2f} | "
            f"COUNTRY={row['country']} | "
            f"VELOCITY={row['velocity_count_5min']} | "
            f"IMPOSSIBLE={row['impossible_travel_flag']} | "
            f"PROBA={row['fraud_probability']:.2%} | "
            f"{row['prediction_label']}"
        )


    # ------------------------------------------------------------------------
    # SAUVEGARDE
    # ------------------------------------------------------------------------

    output_columns = [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "currency",
        "country",
        "merchant_category",
        "device_id",
        "ip_is_datacenter",
        "is_fraud",
        "fraud_type",

        "velocity_count_5min",
        "impossible_travel_flag",

        "fraud_probability",
        "prediction",
        "prediction_label"
    ]


    # conserver seulement les colonnes existantes
    output_columns = [
        c for c in output_columns
        if c in pdf.columns
    ]


    result = pdf[output_columns].copy()


    # fichier unique par micro-batch
    output_file = os.path.join(
        OUTPUT_DIR,
        f"batch_{batch_id}.csv"
    )


    result.to_csv(
        output_file,
        index=False
    )


    # ------------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------------

    fraud_count = (
        result["prediction"] == 1
    ).sum()

    print()
    print(
        f"Transactions traitées : {len(result)}"
    )

    print(
        f"Fraudes détectées : {fraud_count}"
    )

    print(
        f"Normales : {len(result) - fraud_count}"
    )


# ============================================================================
# SPARK
# ============================================================================

print("\n")
print("=" * 70)
print("DEMARRAGE SPARK")
print("=" * 70)


spark = (
    SparkSession.builder
    .appName("RealTimeFraudDetection")
    .master("local[*]")
    .config(
    "spark.jars.packages",
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9"
)
    .config(
        "spark.hadoop.io.native.lib.available",
        "false"
    )
    .config(
        "spark.sql.shuffle.partitions",
        "4"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================================
# SCHEMA KAFKA
# ============================================================================

schema = (
    StructType()
    .add("transaction_id", StringType())
    .add("user_id", StringType())
    .add("timestamp", StringType())
    .add("amount", DoubleType())
    .add("currency", StringType())
    .add("country", StringType())
    .add("merchant_category", StringType())
    .add("device_id", StringType())
    .add("ip_is_datacenter", IntegerType())
    .add("is_fraud", IntegerType())
    .add("fraud_type", StringType())
)


# ============================================================================
# KAFKA
# ============================================================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BROKER
    )
    .option(
        "subscribe",
        TOPIC
    )
    .option(
        "startingOffsets",
        "earliest"
    )
    .load()
)


# ============================================================================
# PARSING JSON
# ============================================================================

transactions = (
    raw_stream
    .select(
        from_json(
            col("value").cast("string"),
            schema
        ).alias("data")
    )
    .select("data.*")
)


# ============================================================================
# NETTOYAGE
# ============================================================================

transactions_clean = (
    transactions
    .filter(
        (col("amount") > 0)
        &
        col("user_id").isNotNull()
        &
        col("transaction_id").isNotNull()
    )
)


# ============================================================================
# FOREACH BATCH
# ============================================================================

query = (
    transactions_clean
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .option(
        "checkpointLocation",
        CHECKPOINT_DIR
    )
    .start()
)


print()
print("=" * 70)
print("🚀 PIPELINE ACTIF")
print("=" * 70)
print("Kafka → Spark → Random Forest → stream_results/")
print()
print("Ctrl+C pour arrêter.")
print("=" * 70)


query.awaitTermination()