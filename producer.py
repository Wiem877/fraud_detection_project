"""
Producteur Kafka - simule l'arrivée en temps réel des transactions
Rejoue transactions_stream.csv ligne par ligne avec un délai entre chaque envoi.
CORRECTION : convertit les champs numériques (amount, ip_is_datacenter, is_fraud)
en vrais types Python avant l'envoi JSON, car csv.DictReader retourne tout en texte.
"""

import csv
import json
import time
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"
CSV_PATH = "fichiers_generes/transactions_stream.csv"
DELAY_SECONDS = 0.3


def convert_types(row):
    """Convertit les champs numériques du texte CSV vers de vrais types Python."""
    row["amount"] = float(row["amount"])
    row["ip_is_datacenter"] = int(row["ip_is_datacenter"])
    row["is_fraud"] = int(row["is_fraud"])
    return row


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    sent = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = convert_types(row)
            producer.send(TOPIC, key=row["user_id"], value=row)
            sent += 1
            if sent % 50 == 0:
                print(f"{sent} transactions envoyées...")
            time.sleep(DELAY_SECONDS)

    producer.flush()
    print(f"Terminé. {sent} transactions envoyées au topic '{TOPIC}'.")


if __name__ == "__main__":
    main()