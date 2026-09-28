"""
Job Spark Structured Streaming - Étape 3 : détection de vélocité
Ajoute une agrégation par fenêtre glissante pour détecter les rafales
de transactions suspectes (plusieurs transactions rapprochées du même
utilisateur), en plus des features simples de l'étape précédente.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col, hour, when, window, expr

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + "\\bin;" + os.environ["PATH"]
os.environ["HADOOP_USER_NAME"] = "dadiw"

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

# ---------------------------------------------------------------------------
# 1. Session Spark
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FraudDetectionVelocity") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.sql.streaming.checkpointLocation", "C:/spark_checkpoints") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Schéma
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
# 3. Lecture Kafka + parsing + nettoyage
# ---------------------------------------------------------------------------
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()

# convertir les données binaires brutes envoyées par kafka en json
transactions = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# nettoyage: rejeter les montants invalides, les users nuls, id transactions nul
transactions_clean = transactions.filter(
    (col("amount") > 0) &
    col("user_id").isNotNull() &
    col("transaction_id").isNotNull()
).withColumn("timestamp_ts", col("timestamp").cast("timestamp"))

# ---------------------------------------------------------------------------
# 4. Détection de vélocité par fenêtre glissante
# ---------------------------------------------------------------------------
# Compte les transactions par utilisateur, par fenêtre de 5 minutes
velocity_counts = transactions_clean \
    .withWatermark("timestamp_ts", "10 minutes") \
    .groupBy(
        window(col("timestamp_ts"), "5 minutes", "1 minute"),
        col("user_id")
    ) \
    .count()

# On ne garde que les cas suspects : 5 transactions ou plus dans la fenêtre
velocity_alerts = velocity_counts.filter(col("count") >= 5) \
    .selectExpr(
        "user_id",
        "window.start as window_start",
        "window.end as window_end",
        "count as nb_transactions"
    )

# ---------------------------------------------------------------------------
# 4bis. Détection de voyage impossible (self-join stream-stream)
# ---------------------------------------------------------------------------
# On compare chaque transaction à toutes les autres transactions DU MEME
# utilisateur qui arrivent dans les 20 minutes suivantes, dans un pays
# différent. Les deux "copies" du flux ont besoin de leur propre watermark.
tx_a = transactions_clean.withWatermark("timestamp_ts", "10 minutes").alias("a")
tx_b = transactions_clean.withWatermark("timestamp_ts", "10 minutes").alias("b")

impossible_travel_pairs = tx_a.join(
    tx_b,
    expr("""
        a.user_id = b.user_id AND
        a.transaction_id != b.transaction_id AND
        b.timestamp_ts > a.timestamp_ts AND
        b.timestamp_ts <= a.timestamp_ts + interval 20 minutes AND
        a.country != b.country
    """)
)

impossible_travel_alerts = impossible_travel_pairs.select(
    col("a.user_id").alias("user_id"),
    col("a.transaction_id").alias("transaction_1"),
    col("a.country").alias("country_1"),
    col("a.timestamp_ts").alias("time_1"),
    col("b.transaction_id").alias("transaction_2"),
    col("b.country").alias("country_2"),
    col("b.timestamp_ts").alias("time_2")
)

# ---------------------------------------------------------------------------
# 5. Ajouter de nouvelles colonnes calculées (features simples)
# ---------------------------------------------------------------------------
transactions_features = transactions_clean \
    .withColumn("hour_of_day", hour(col("timestamp_ts"))) \
    .withColumn(
        "is_night",
        when((col("hour_of_day") >= 0) & (col("hour_of_day") < 5), True).otherwise(False)
    )

# Affiche transactions_features en continu dans la console
query_features = transactions_features.select(
    "transaction_id", "user_id", "amount", "country", "hour_of_day", "is_night", "is_fraud"
).writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .queryName("features") \
    .start()

# Affiche velocity_alerts dans la console, mis à jour à chaque changement
query_velocity = velocity_alerts.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .queryName("velocity_alerts") \
    .start()

# Affiche impossible_travel_alerts dès qu'une paire suspecte est détectée
query_impossible_travel = impossible_travel_alerts.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .queryName("impossible_travel_alerts") \
    .start()

print("Spark détecte les features simples, la vélocité ET le voyage impossible en parallèle... (Ctrl+C pour arrêter)")
spark.streams.awaitAnyTermination()