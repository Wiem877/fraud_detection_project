"""
Consommateur de test - vérifie que les messages arrivent bien dans Kafka.
Utile pour valider le pipeline avant de brancher Spark à l'étape suivante.
"""

import json
from     import KafkaConsumer

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="test-consumer-group",
)

print("En attente de messages... (Ctrl+C pour arrêter)")
for message in consumer:
    tx = message.value
    flag = "🚨 FRAUDE" if tx["is_fraud"] == "1" else "OK"
    print(f"[{flag}] {tx['transaction_id']} | user={tx['user_id']} | "
          f"amount={tx['amount']} | country={tx['country']}")
