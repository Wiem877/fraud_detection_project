"""
Générateur de données de transactions bancaires labellisées (fraude/normal)
Projet : Détection de fraude en temps réel - Banque tunisienne (simulation)
"""

import random
import uuid # génération d'identifiants uniques
import csv
from datetime import datetime, timedelta

random.seed(42) # garder les memes valeurs random à chaque execution, les données sont regénrées mais seed fixe le générateur toujours sur 42 donc les valeurs restent les meme

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
NUM_USERS = 500 #nombre des personnages simulés
DAYS = 30
START_DATE = datetime(2026, 7, 1)
FRAUD_RATIO = 0.02          # ~2% de transactions frauduleuses (réaliste)
AVG_TX_PER_USER_PER_DAY = 3

COUNTRIES = ["TN", "FR", "DE", "TH", "NG", "AE", "IT", "US", "TR"]
HOME_COUNTRY_WEIGHTS = {"TN": 0.90, "FR": 0.03, "DE": 0.02, "IT": 0.02,
                         "AE": 0.01, "TR": 0.01, "US": 0.005, "TH": 0.005, "NG": 0.005} #90% d'un user tunisien est un client dans banque tunisienne ...
# Pays où le taux de fraude interne observé est plus élevé (simulé, pas une vérité universelle)
HIGH_RISK_COUNTRIES = {"TH", "NG"}

MERCHANT_CATEGORIES = ["grocery", "electronics", "restaurant", "travel", #catégories des commerçants chez lesquels l'utilisateur peut effectuer une transaction
                        "clothing", "online_services", "fuel", "pharmacy"]

FRAUD_TYPES = ["velocity", "impossible_travel", "high_amount",
               "new_device_vpn", "night_risk_country"]


# ---------------------------------------------------------------------------
# PROFILS UTILISATEURS
# ---------------------------------------------------------------------------
def build_user_profiles(n):
    profiles = {}
    for i in range(n):
        user_id = f"U{i:05d}"
        home_country = random.choices(list(HOME_COUNTRY_WEIGHTS.keys()),
                                       weights=list(HOME_COUNTRY_WEIGHTS.values()))[0]
        profiles[user_id] = {
            "home_country": home_country,
            "avg_amount": round(random.uniform(30, 400), 2),
            "usual_devices": [f"D{uuid.uuid4().hex[:8]}" for _ in range(random.randint(1, 2))],
            "usual_hour_start": random.randint(6, 10),
            "usual_hour_end": random.randint(18, 23),
            "usual_categories": random.sample(MERCHANT_CATEGORIES, k=random.randint(2, 4)),
        }
    return profiles


def new_tx_id():
    return f"TXN{uuid.uuid4().hex[:10]}"


def base_row(user_id, ts, amount, country, category, device_id,
             ip_is_datacenter, is_fraud, fraud_type):
    return {
        "transaction_id": new_tx_id(),
        "user_id": user_id,
        "timestamp": ts.isoformat(),
        "amount": round(max(amount, 1.0), 2),
        "currency": "TND",
        "country": country,
        "merchant_category": category,
        "device_id": device_id,
        "ip_is_datacenter": int(ip_is_datacenter),
        "is_fraud": int(is_fraud),
        "fraud_type": fraud_type,
    }


# ---------------------------------------------------------------------------
# TRANSACTIONS NORMALES (avec un peu de bruit volontaire)
# ---------------------------------------------------------------------------
def generate_normal_transaction(user_id, profile, ts):
    amount = random.gauss(profile["avg_amount"], profile["avg_amount"] * 0.25)
    country = profile["home_country"]
    # bruit : 3% du temps un pays différent mais légitime (vrai voyage)
    if random.random() < 0.03:
        country = random.choice(COUNTRIES)
    device = random.choice(profile["usual_devices"])
    # bruit : 2% du temps un nouvel appareil (ex: nouveau téléphone), reste normal
    if random.random() < 0.02:
        device = f"D{uuid.uuid4().hex[:8]}"
    category = random.choice(profile["usual_categories"] + MERCHANT_CATEGORIES[:2])
    ip_dc = random.random() < 0.01  # bruit léger : rare faux positif possible
    return base_row(user_id, ts, amount, country, category, device, ip_dc, False, "none")


# ---------------------------------------------------------------------------
# SCÉNARIOS DE FRAUDE (avec variance pour éviter les seuils triviaux)
# ---------------------------------------------------------------------------
def generate_fraud_burst(user_id, profile, ts, fraud_type):
    rows = []

    if fraud_type == "velocity":
        n = random.randint(5, 9)
        t = ts
        for _ in range(n):
            amount = random.gauss(profile["avg_amount"] * 0.8, profile["avg_amount"] * 0.3)
            device = random.choice(profile["usual_devices"] + [f"D{uuid.uuid4().hex[:8]}"])
            rows.append(base_row(user_id, t, amount, profile["home_country"],
                                  random.choice(MERCHANT_CATEGORIES), device,
                                  random.random() < 0.2, True, fraud_type))
            t += timedelta(seconds=random.randint(5, 90))

    elif fraud_type == "impossible_travel":
        c1 = profile["home_country"]
        c2 = random.choice([c for c in COUNTRIES if c != c1])
        amount1 = random.gauss(profile["avg_amount"], profile["avg_amount"] * 0.3)
        amount2 = random.gauss(profile["avg_amount"] * 1.5, profile["avg_amount"] * 0.4)
        rows.append(base_row(user_id, ts, amount1, c1, random.choice(MERCHANT_CATEGORIES),
                              random.choice(profile["usual_devices"]), False, True, fraud_type))
        t2 = ts + timedelta(minutes=random.randint(2, 20))  # trop court pour un vrai trajet
        rows.append(base_row(user_id, t2, amount2, c2, random.choice(MERCHANT_CATEGORIES),
                              f"D{uuid.uuid4().hex[:8]}", True, True, fraud_type))

    elif fraud_type == "high_amount":
        amount = profile["avg_amount"] * random.uniform(6, 14)
        rows.append(base_row(user_id, ts, amount, profile["home_country"],
                              random.choice(MERCHANT_CATEGORIES),
                              random.choice(profile["usual_devices"]),
                              random.random() < 0.3, True, fraud_type))

    elif fraud_type == "new_device_vpn":
        # montant proche de la normale : signal caché dans device+IP, pas le montant
        amount = random.gauss(profile["avg_amount"], profile["avg_amount"] * 0.2)
        rows.append(base_row(user_id, ts, amount, profile["home_country"],
                              random.choice(profile["usual_categories"]),
                              f"D{uuid.uuid4().hex[:8]}", True, True, fraud_type))

    elif fraud_type == "night_risk_country":
        night_ts = ts.replace(hour=random.randint(2, 5), minute=random.randint(0, 59))
        amount = random.gauss(profile["avg_amount"], profile["avg_amount"] * 0.4)
        rows.append(base_row(user_id, night_ts, amount, random.choice(list(HIGH_RISK_COUNTRIES)),
                              random.choice(MERCHANT_CATEGORIES),
                              f"D{uuid.uuid4().hex[:8]}", random.random() < 0.5, True, fraud_type))

    return rows


# ---------------------------------------------------------------------------
# GÉNÉRATION COMPLÈTE
# ---------------------------------------------------------------------------
def generate_dataset():
    profiles = build_user_profiles(NUM_USERS)
    all_rows = []

    for user_id, profile in profiles.items():
        n_tx = int(AVG_TX_PER_USER_PER_DAY * DAYS * random.uniform(0.6, 1.4))
        for _ in range(n_tx):
            day_offset = random.randint(0, DAYS - 1)
            hour = random.randint(profile["usual_hour_start"], profile["usual_hour_end"])
            ts = START_DATE + timedelta(days=day_offset, hours=hour,
                                         minutes=random.randint(0, 59))
            if random.random() < FRAUD_RATIO:
                fraud_type = random.choice(FRAUD_TYPES)
                all_rows.extend(generate_fraud_burst(user_id, profile, ts, fraud_type))
            else:
                all_rows.append(generate_normal_transaction(user_id, profile, ts))

    all_rows.sort(key=lambda r: r["timestamp"])
    return all_rows


def split_by_time(rows, train_ratio=0.7, test_ratio=0.2):
    n = len(rows)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))
    return rows[:train_end], rows[train_end:test_end], rows[test_end:]


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    data = generate_dataset()
    train, test, stream = split_by_time(data)

write_csv(train, r"C:\Users\dadiw\OneDrive\Desktop\fraud_detection_project\fichiers_generes\transactions_train.csv")
write_csv(test, r"C:\Users\dadiw\OneDrive\Desktop\fraud_detection_project\fichiers_generes\transactions_test.csv")
write_csv(stream, r"C:\Users\dadiw\OneDrive\Desktop\fraud_detection_project\fichiers_generes\transactions_stream.csv")

total = len(data)
n_fraud = sum(r["is_fraud"] for r in data)
print(f"Total transactions générées : {total}")
print(f"Dont frauduleuses : {n_fraud} ({100 * n_fraud / total:.2f}%)")
print(f"Train : {len(train)} | Test : {len(test)} | Stream (simulation Kafka) : {len(stream)}")
print("\nRépartition des types de fraude :")
from collections import Counter
print(Counter(r["fraud_type"] for r in data if r["is_fraud"]))