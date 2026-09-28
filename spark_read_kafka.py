"""
Job Spark Structured Streaming - Étape 1 : lecture simple du flux Kafka
Objectif : valider que Spark peut lire les transactions depuis Kafka,
avant d'ajouter le nettoyage et les features à l'étape suivante.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

# ---------------------------------------------------------------------------
# 1. Démarrage de la session Spark
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FraudDetectionStreamReader") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")  # réduit les logs Spark verbeux dans la console

# ---------------------------------------------------------------------------
# 2. Définir la structure (schéma) attendue de chaque transaction
# ---------------------------------------------------------------------------
# Doit correspondre exactement aux colonnes générées par generate_data.py
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
# 3. Lecture du flux Kafka en continu
# ---------------------------------------------------------------------------
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()

# Kafka envoie les données en binaire (colonne "value") -> on les convertit
# en texte JSON, puis on les transforme en colonnes structurées via le schéma
transactions = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# ---------------------------------------------------------------------------
# 4. Affichage du flux dans la console (juste pour valider que ça marche)
# ---------------------------------------------------------------------------
query = transactions.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

print("Spark écoute le topic Kafka 'transactions'... (Ctrl+C pour arrêter)")
query.awaitTermination()
