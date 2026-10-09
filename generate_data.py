"""
Générateur V2 de données de transactions bancaires
Projet : Détection de fraude en temps réel - Banque tunisienne (simulation)

V2 :
- Risque cyber basé sur le World Cybercrime Index
- Risque AML/FATF
- Fraudes multi-signaux
- Variabilité des scénarios
- Transactions normales dans les pays à risque
"""

import random
import uuid
import csv
from datetime import datetime, timedelta
from collections import Counter

random.seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_USERS = 500
DAYS = 30
START_DATE = datetime(2026, 7, 1)

# Probabilité de déclencher un scénario frauduleux
FRAUD_RATIO = 0.02

AVG_TX_PER_USER_PER_DAY = 3


# ============================================================================
# PAYS
# ============================================================================

COUNTRIES = [
    "TN", "FR", "DE", "IT", "AE", "TR", "US",
    "RU", "UA", "CN", "NG", "RO", "KP",
    "BR", "IN", "IR", "MM",
    "LB", "VE", "VN", "SY", "KE", "CI", "BO"
]


# Pays d'origine avec forte activité cybercriminelle
# Source : World Cybercrime Index 2024
CYBERCRIME_RISK_COUNTRIES = {
    "RU",   # Russie
    "UA",   # Ukraine
    "CN",   # Chine
    "US",   # États-Unis
    "NG",   # Nigéria
    "RO",   # Roumanie
    "BR",   # Brésil
}


# Score simplifié inspiré du World Cybercrime Index.
# IMPORTANT :
# ce score est utilisé comme signal de simulation,
# pas comme probabilité réelle de fraude.
WORLD_CYBERCRIME_SCORE = {
    "RU": 8.96,
    "UA": 8.37,
    "CN": 8.22,
    "US": 7.99,
    "NG": 8.25,
    "RO": 7.12,
    "BR": 6.90,
}


# Juridictions FATF actuellement "High-Risk Jurisdictions
# subject to a Call for Action".
# Situation vérifiée en 2026.
FATF_BLACKLIST = {
    "KP",   # Corée du Nord
    "IR",   # Iran
    "MM",   # Myanmar
}


# Juridictions actuellement sous "Increased Monitoring".
FATF_GREYLIST = {
    "DZ",   # Algérie
    "AO",   # Angola
    "BO",   # Bolivie
    "BA",   # Bosnie-Herzégovine
    "BG",   # Bulgarie
    "CM",   # Cameroun
    "CI",   # Côte d'Ivoire
    "CD",   # RDC
    "HT",   # Haïti
    "IQ",   # Irak
    "KE",   # Kenya
    "KW",   # Koweït
    "LA",   # Laos
    "LB",   # Liban
    "MC",   # Monaco
    "NP",   # Népal
    "PG",   # Papouasie-Nouvelle-Guinée
    "SS",   # Soudan du Sud
    "SY",   # Syrie
    "VE",   # Venezuela
    "VN",   # Vietnam
    "YE",   # Yémen
}


# ============================================================================
# PROFILS UTILISATEURS
# ============================================================================

HOME_COUNTRY_WEIGHTS = {
    "TN": 0.90,
    "FR": 0.025,
    "DE": 0.015,
    "IT": 0.015,
    "AE": 0.01,
    "TR": 0.01,
    "US": 0.005,
    "RU": 0.002,
    "UA": 0.002,
    "CN": 0.002,
    "NG": 0.002,
    "RO": 0.001,
    "BR": 0.001,
    "IN": 0.002,
    "IR": 0.001,
    "KP": 0.0002,
    "MM": 0.0002,
    "LB": 0.001,
    "VE": 0.0002,
    "VN": 0.0005,
    "SY": 0.0002,
    "KE": 0.0005,
    "CI": 0.0005,
    "BO": 0.0002,
}


MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "restaurant",
    "travel",
    "clothing",
    "online_services",
    "fuel",
    "pharmacy"
]


FRAUD_TYPES = [
    "velocity",
    "impossible_travel",
    "high_amount",
    "new_device_vpn",
    "night_risk_country",
    "multi_signal"
]


# ============================================================================
# UTILITAIRES RISQUE PAYS
# ============================================================================

def get_country_risk_score(country):
    """
    Calcule un score de risque simulé à partir de deux sources :
    - World Cybercrime Index
    - classification FATF
    """

    cyber_score = WORLD_CYBERCRIME_SCORE.get(country, 0)

    # Normalisation du score cyber sur environ 0-1
    cyber_risk = cyber_score / 10

    # Risque FATF
    if country in FATF_BLACKLIST:
        fatf_risk = 1.0
    elif country in FATF_GREYLIST:
        fatf_risk = 0.5
    else:
        fatf_risk = 0.0

    # Combinaison
    risk_score = (
        0.6 * cyber_risk +
        0.4 * fatf_risk
    )

    return round(risk_score, 3)


# ============================================================================
# PROFILS
# ============================================================================

def build_user_profiles(n):

    profiles = {}

    for i in range(n):

        user_id = f"U{i:05d}"

        home_country = random.choices(
            list(HOME_COUNTRY_WEIGHTS.keys()),
            weights=list(HOME_COUNTRY_WEIGHTS.values())
        )[0]

        profiles[user_id] = {

            "home_country": home_country,

            "avg_amount": round(
                random.uniform(30, 400),
                2
            ),

            "usual_devices": [
                f"D{uuid.uuid4().hex[:8]}"
                for _ in range(random.randint(1, 2))
            ],

            "usual_hour_start": random.randint(6, 10),

            "usual_hour_end": random.randint(18, 23),

            "usual_categories": random.sample(
                MERCHANT_CATEGORIES,
                k=random.randint(2, 4)
            ),
        }

    return profiles


# ============================================================================
# ID TRANSACTION
# ============================================================================

def new_tx_id():

    return f"TXN{uuid.uuid4().hex[:10]}"


# ============================================================================
# CRÉATION D'UNE LIGNE
# ============================================================================

def base_row(
    user_id,
    ts,
    amount,
    country,
    category,
    device_id,
    ip_is_datacenter,
    is_fraud,
    fraud_type
):

    cyber_score = WORLD_CYBERCRIME_SCORE.get(
        country,
        0
    )

    return {

        "transaction_id":
            new_tx_id(),

        "user_id":
            user_id,

        "timestamp":
            ts.isoformat(),

        "amount":
            round(max(amount, 1.0), 2),

        "currency":
            "TND",

        "country":
            country,

        "merchant_category":
            category,

        "device_id":
            device_id,

        "ip_is_datacenter":
            int(ip_is_datacenter),

        # ----------------------------
        # NOUVELLES FEATURES V2
        # ----------------------------

        "is_cybercrime_risk":
            int(country in CYBERCRIME_RISK_COUNTRIES),

        "world_cybercrime_score":
            cyber_score,

        "is_fatf_blacklist":
            int(country in FATF_BLACKLIST),

        "is_fatf_greylist":
            int(country in FATF_GREYLIST),

        "country_risk_score":
            get_country_risk_score(country),

        # ----------------------------

        "is_fraud":
            int(is_fraud),

        "fraud_type":
            fraud_type,
    }


# ============================================================================
# TRANSACTION NORMALE
# ============================================================================

def generate_normal_transaction(
    user_id,
    profile,
    ts
):

    # Montant proche de l'habitude
    amount = random.gauss(
        profile["avg_amount"],
        profile["avg_amount"] * 0.25
    )

    country = profile["home_country"]

    # ---------------------------------------------------------
    # Bruit réaliste : voyage à l'étranger
    # ---------------------------------------------------------

    if random.random() < 0.05:

        country = random.choice(COUNTRIES)

    # ---------------------------------------------------------
    # Nouveau device légitime
    # ---------------------------------------------------------

    device = random.choice(
        profile["usual_devices"]
    )

    if random.random() < 0.03:

        device = f"D{uuid.uuid4().hex[:8]}"

    # ---------------------------------------------------------
    # Catégorie
    # ---------------------------------------------------------

    category = random.choice(
        profile["usual_categories"]
        + MERCHANT_CATEGORIES[:2]
    )

    # ---------------------------------------------------------
    # IP datacenter rare mais possible
    # ---------------------------------------------------------

    ip_dc = random.random() < 0.01

    return base_row(

        user_id=user_id,

        ts=ts,

        amount=amount,

        country=country,

        category=category,

        device_id=device,

        ip_is_datacenter=ip_dc,

        is_fraud=False,

        fraud_type="none"
    )


# ============================================================================
# FRAUD BURST
# ============================================================================

def generate_fraud_burst(
    user_id,
    profile,
    ts,
    fraud_type
):

    rows = []


    # ========================================================================
    # 1. VELOCITY
    # ========================================================================

    if fraud_type == "velocity":

        n = random.randint(5, 9)

        t = ts

        for _ in range(n):

            amount = random.gauss(
                profile["avg_amount"] * 0.8,
                profile["avg_amount"] * 0.3
            )

            device = random.choice(
                profile["usual_devices"]
                + [f"D{uuid.uuid4().hex[:8]}"]
            )

            rows.append(

                base_row(

                    user_id,
                    t,
                    amount,
                    profile["home_country"],
                    random.choice(MERCHANT_CATEGORIES),
                    device,
                    random.random() < 0.2,
                    True,
                    fraud_type
                )
            )

            t += timedelta(
                seconds=random.randint(5, 60)
            )


    # ========================================================================
    # 2. IMPOSSIBLE TRAVEL
    # ========================================================================

    elif fraud_type == "impossible_travel":

        c1 = profile["home_country"]

        c2 = random.choice([
            c for c in COUNTRIES
            if c != c1
        ])

        amount1 = random.gauss(
            profile["avg_amount"],
            profile["avg_amount"] * 0.3
        )

        amount2 = random.gauss(
            profile["avg_amount"] * 1.5,
            profile["avg_amount"] * 0.4
        )

        rows.append(

            base_row(
                user_id,
                ts,
                amount1,
                c1,
                random.choice(MERCHANT_CATEGORIES),
                random.choice(profile["usual_devices"]),
                False,
                True,
                fraud_type
            )
        )

        t2 = ts + timedelta(
            minutes=random.randint(2, 20)
        )

        rows.append(

            base_row(
                user_id,
                t2,
                amount2,
                c2,
                random.choice(MERCHANT_CATEGORIES),
                f"D{uuid.uuid4().hex[:8]}",
                True,
                True,
                fraud_type
            )
        )


    # ========================================================================
    # 3. HIGH AMOUNT
    # ========================================================================

    elif fraud_type == "high_amount":

        amount = (
            profile["avg_amount"]
            * random.uniform(6, 14)
        )

        rows.append(

            base_row(
                user_id,
                ts,
                amount,
                profile["home_country"],
                random.choice(MERCHANT_CATEGORIES),
                random.choice(profile["usual_devices"]),
                random.random() < 0.3,
                True,
                fraud_type
            )
        )


    # ========================================================================
    # 4. NEW DEVICE + VPN
    # ========================================================================

    elif fraud_type == "new_device_vpn":

        amount = random.gauss(
            profile["avg_amount"],
            profile["avg_amount"] * 0.2
        )

        rows.append(

            base_row(
                user_id,
                ts,
                amount,
                profile["home_country"],
                random.choice(profile["usual_categories"]),
                f"D{uuid.uuid4().hex[:8]}",
                True,
                True,
                fraud_type
            )
        )


    # ========================================================================
    # 5. NIGHT + COUNTRY À RISQUE
    # ========================================================================

    elif fraud_type == "night_risk_country":

        night_ts = ts.replace(

            hour=random.randint(1, 5),

            minute=random.randint(0, 59)
        )

        amount = random.gauss(
            profile["avg_amount"],
            profile["avg_amount"] * 0.4
        )

        # Ici on choisit un pays ayant un signal cyber
        # ou FATF.
        risky_countries = list(
            CYBERCRIME_RISK_COUNTRIES
            | FATF_BLACKLIST
            | FATF_GREYLIST
        )

        country = random.choice(
            risky_countries
        )

        rows.append(

            base_row(
                user_id,
                night_ts,
                amount,
                country,
                random.choice(MERCHANT_CATEGORIES),
                f"D{uuid.uuid4().hex[:8]}",
                random.random() < 0.5,
                True,
                fraud_type
            )
        )


    # ========================================================================
    # 6. MULTI-SIGNAL FRAUD - NOUVEAU V2
    # ========================================================================

    elif fraud_type == "multi_signal":

        # ------------------------------------------------------------
        # Pays présentant un signal cyber ou AML
        # ------------------------------------------------------------

        risky_countries = list(
            CYBERCRIME_RISK_COUNTRIES
            | FATF_BLACKLIST
            | FATF_GREYLIST
        )

        country = random.choice(
            risky_countries
        )

        # ------------------------------------------------------------
        # Heure inhabituelle
        # ------------------------------------------------------------

        fraud_hour = random.randint(0, 5)

        fraud_minute = random.randint(0, 59)

        fraud_ts = ts.replace(
            hour=fraud_hour,
            minute=fraud_minute
        )

        # ------------------------------------------------------------
        # Montant anormal
        # ------------------------------------------------------------

        amount = (
            profile["avg_amount"]
            * random.uniform(2.5, 7)
        )

        # ------------------------------------------------------------
        # Nouveau device
        # ------------------------------------------------------------

        new_device = (
            f"D{uuid.uuid4().hex[:8]}"
        )

        # ------------------------------------------------------------
        # Catégorie inhabituelle
        # ------------------------------------------------------------

        unusual_categories = [
            c for c in MERCHANT_CATEGORIES
            if c not in profile["usual_categories"]
        ]

        if unusual_categories:

            category = random.choice(
                unusual_categories
            )

        else:

            category = random.choice(
                MERCHANT_CATEGORIES
            )

        # ------------------------------------------------------------
        # IP datacenter / VPN
        # ------------------------------------------------------------

        ip_dc = random.random() < 0.85

        # ------------------------------------------------------------
        # Première transaction
        # ------------------------------------------------------------

        rows.append(

            base_row(

                user_id=user_id,

                ts=fraud_ts,

                amount=amount,

                country=country,

                category=category,

                device_id=new_device,

                ip_is_datacenter=ip_dc,

                is_fraud=True,

                fraud_type=fraud_type
            )
        )

        # ------------------------------------------------------------
        # Quelques transactions très rapprochées
        # ------------------------------------------------------------

        n_extra = random.randint(2, 4)

        current_ts = fraud_ts

        for _ in range(n_extra):

            current_ts += timedelta(
                seconds=random.randint(20, 90)
            )

            extra_amount = (
                profile["avg_amount"]
                * random.uniform(1.5, 5)
            )

            rows.append(

                base_row(

                    user_id=user_id,

                    ts=current_ts,

                    amount=extra_amount,

                    country=country,

                    category=random.choice(
                        MERCHANT_CATEGORIES
                    ),

                    device_id=new_device,

                    ip_is_datacenter=
                        random.random() < 0.9,

                    is_fraud=True,

                    fraud_type=fraud_type
                )
            )


    return rows


# ============================================================================
# GENERATION DU DATASET
# ============================================================================

def generate_dataset():

    profiles = build_user_profiles(
        NUM_USERS
    )

    all_rows = []

    for user_id, profile in profiles.items():

        n_tx = int(
            AVG_TX_PER_USER_PER_DAY
            * DAYS
            * random.uniform(0.6, 1.4)
        )

        for _ in range(n_tx):

            day_offset = random.randint(
                0,
                DAYS - 1
            )

            hour = random.randint(
                profile["usual_hour_start"],
                profile["usual_hour_end"]
            )

            ts = START_DATE + timedelta(

                days=day_offset,

                hours=hour,

                minutes=random.randint(0, 59)
            )

            # ----------------------------------------------------
            # Transaction frauduleuse ?
            # ----------------------------------------------------

            if random.random() < FRAUD_RATIO:

                fraud_type = random.choice(
                    FRAUD_TYPES
                )

                all_rows.extend(

                    generate_fraud_burst(
                        user_id,
                        profile,
                        ts,
                        fraud_type
                    )
                )

            else:

                all_rows.append(

                    generate_normal_transaction(
                        user_id,
                        profile,
                        ts
                    )
                )


    # ------------------------------------------------------------
    # Tri chronologique
    # ------------------------------------------------------------

    all_rows.sort(
        key=lambda r: r["timestamp"]
    )

    return all_rows


# ============================================================================
# SPLIT TEMPOREL
# ============================================================================

def split_by_time(
    rows,
    train_ratio=0.7,
    test_ratio=0.2
):

    n = len(rows)

    train_end = int(
        n * train_ratio
    )

    test_end = int(
        n * (train_ratio + test_ratio)
    )

    return (
        rows[:train_end],
        rows[train_end:test_end],
        rows[test_end:]
    )


# ============================================================================
# CSV
# ============================================================================

def write_csv(rows, path):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    data = generate_dataset()

    train, test, stream = split_by_time(data)


    OUTPUT_DIR = (
        r"C:\Users\dadiw\OneDrive\Desktop"
        r"\fraud_detection_project"
        r"\fichiers_generes"
    )


    write_csv(
        train,
        OUTPUT_DIR + r"\transactions_train.csv"
    )

    write_csv(
        test,
        OUTPUT_DIR + r"\transactions_test.csv"
    )

    write_csv(
        stream,
        OUTPUT_DIR + r"\transactions_stream.csv"
    )


    # ========================================================================
    # STATISTIQUES
    # ========================================================================

    total = len(data)

    n_fraud = sum(
        r["is_fraud"]
        for r in data
    )


    print(
        f"Total transactions générées : {total}"
    )

    print(
        f"Dont frauduleuses : "
        f"{n_fraud} "
        f"({100 * n_fraud / total:.2f}%)"
    )

    print(
        f"Train : {len(train)} | "
        f"Test : {len(test)} | "
        f"Stream : {len(stream)}"
    )


    print(
        "\nRépartition des types de fraude :"
    )

    print(
        Counter(
            r["fraud_type"]
            for r in data
            if r["is_fraud"]
        )
    )


    print(
        "\nRépartition des pays :"
    )

    print(
        Counter(
            r["country"]
            for r in data
        )
    )


    print(
        "\nFraudes par pays :"
    )

    print(
        Counter(
            r["country"]
            for r in data
            if r["is_fraud"]
        )
    )