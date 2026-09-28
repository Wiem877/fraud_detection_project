"""
Script de DIAGNOSTIC uniquement - affiche les transactions SANS filtre
ni conversion, pour voir exactement ce que Spark reçoit et comment
il interprète chaque colonne (types, valeurs nulles...).
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col

os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + "\\bin;" + os.environ["PATH"]
os.environ["HADOOP_USER_NAME"] = "dadiw"

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

spark = SparkSession.builder \
    .appName("Diagnostic") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.sql.streaming.checkpointLocation", "C:/spark_checkpoints_debug") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

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

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# ÉTAPE A : afficher le JSON brut, tel que reçu, sans aucun parsing
raw_text = raw_stream.selectExpr("CAST(value AS STRING) as json_brut")

query_raw = raw_text.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .queryName("raw_json") \
    .start()

# ÉTAPE B : afficher après parsing avec le schéma (sans filtre)
transactions = raw_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

query_parsed = transactions.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .queryName("parsed") \
    .start()

print("DIAGNOSTIC : compare le JSON brut (raw_json) et le résultat parsé (parsed)")
print("Lance producer.py maintenant dans l'autre terminal.")
spark.streams.awaitAnyTermination()