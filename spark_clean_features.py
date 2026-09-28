"""
Job Spark Structured Streaming - Étape 2 : nettoyage + premières features
Ajoute le nettoyage des données et des indicateurs simples (calculables
transaction par transaction, sans historique).
Les features nécessitant l'historique (vélocité, voyage impossible)
seront ajoutées à l'étape suivante, avec les fenêtres temporelles.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col, hour, when

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + "\\bin;" + os.environ["PATH"]
os.environ["HADOOP_USER_NAME"] = "dadiw"

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

# ---------------------------------------------------------------------------
# 1. Session Spark
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FraudDetectionCleanFeatures") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.sql.streaming.checkpointLocation", "C:/spark_checkpoints") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Schéma des transactions (identique à l'étape précédente)
# ---------------------------------------------------------------------------
schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("user_id", StringType()) \
    .add("timestamp", StringType()) \
    .add("amount", DoubleType()) \
    .add("currency", StringType()) \
    .add("country", StringType()) \
    .add("merchant_category", StringType()) \
    .add("device_id", StringType()) \
    .add("ip_is_datacenter", IntegerType()) \
    .add("is_fraud", IntegerType()) \
    .add("fraud_type", StringType())

# ---------------------------------------------------------------------------
# 3. Lecture Kafka + parsing JSON (identique à l'étape précédente)
# ---------------------------------------------------------------------------
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()

transactions = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# ---------------------------------------------------------------------------
# 4. NETTOYAGE : on rejette les lignes invalides
# ---------------------------------------------------------------------------
transactions_clean = transactions.filter(
    (col("amount") > 0) &
    col("user_id").isNotNull() &
    col("transaction_id").isNotNull()
)

# ---------------------------------------------------------------------------
# 5. FEATURES SIMPLES : calculables ligne par ligne, sans historique
# ---------------------------------------------------------------------------
# Convertit la colonne timestamp (texte) en vrai type "timestamp" Spark,
# puis extrait l'heure de la journée
transactions_features = transactions_clean \
    .withColumn("timestamp_ts", col("timestamp").cast("timestamp")) \
    .withColumn("hour_of_day", hour(col("timestamp_ts"))) \
    .withColumn(
        "is_night",
        when((col("hour_of_day") >= 0) & (col("hour_of_day") < 5), True).otherwise(False)
    ) \
    .withColumn(
        "amount_category",
        when(col("amount") < 100, "low")
        .when(col("amount") < 500, "medium")
        .otherwise("high")
    )

# ---------------------------------------------------------------------------
# 6. Affichage du résultat enrichi dans la console
# ---------------------------------------------------------------------------
output_columns = transactions_features.select(
    "transaction_id", "user_id", "amount", "amount_category",
    "country", "hour_of_day", "is_night", "ip_is_datacenter", "is_fraud"
)

query = output_columns.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

print("Spark nettoie et enrichit le flux Kafka 'transactions'... (Ctrl+C pour arrêter)")
query.awaitTermination()
