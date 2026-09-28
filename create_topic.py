"""
Crée explicitement le topic Kafka 'transactions' AVANT de lancer producer.py
ou spark_velocity.py. Élimine la race condition où plusieurs processus
tentent de créer le topic en même temps au démarrage.
À lancer UNE SEULE FOIS après chaque `docker compose up -d`.
"""

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

KAFKA_BROKER = "localhost:29092"
TOPIC = "transactions"

admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER, client_id="setup-admin")

try:
    admin.create_topics([NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)])
    print(f"Topic '{TOPIC}' créé avec succès.")
except TopicAlreadyExistsError:
    print(f"Topic '{TOPIC}' existe déjà, rien à faire.")
finally:
    admin.close()