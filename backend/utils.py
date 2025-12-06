import hashlib
import os
from datetime import datetime

USERS = {
    "ahmed": "ahmed123",
    "bilal": "bilal123",
    "sana": "sana123",
    "fatima": "fatima123",
    "omar": "omar123",
    "ayesha": "ayesha123",
    "hassan": "hassan123",
    "zainab": "zainab123",
    "ali": "ali123",
    "maryam": "maryam123",
    "yusuf": "yusuf123",
    "khadija": "khadija123",
    "ibrahim": "ibrahim123",
    "noor": "noor123",
    "hamza": "hamza123",
    "sara": "sara123",
    "tariq": "tariq123",
    "amina": "amina123",
    "khalid": "khalid123",
    "layla": "layla123"
}

USER_DATA = {
    "ahmed": {"balance": 25000.00, "user_id": "USR001"},
    "bilal": {"balance": 18500.00, "user_id": "USR002"},
    "sana": {"balance": 32000.00, "user_id": "USR003"},
    "fatima": {"balance": 15000.00, "user_id": "USR004"},
    "omar": {"balance": 42000.00, "user_id": "USR005"},
    "ayesha": {"balance": 28000.00, "user_id": "USR006"},
    "hassan": {"balance": 19500.00, "user_id": "USR007"},
    "zainab": {"balance": 36000.00, "user_id": "USR008"},
    "ali": {"balance": 22000.00, "user_id": "USR009"},
    "maryam": {"balance": 31000.00, "user_id": "USR010"},
    "yusuf": {"balance": 27500.00, "user_id": "USR011"},
    "khadija": {"balance": 20000.00, "user_id": "USR012"},
    "ibrahim": {"balance": 45000.00, "user_id": "USR013"},
    "noor": {"balance": 16500.00, "user_id": "USR014"},
    "hamza": {"balance": 38000.00, "user_id": "USR015"},
    "sara": {"balance": 24000.00, "user_id": "USR016"},
    "tariq": {"balance": 29500.00, "user_id": "USR017"},
    "amina": {"balance": 21000.00, "user_id": "USR018"},
    "khalid": {"balance": 33500.00, "user_id": "USR019"},
    "layla": {"balance": 26000.00, "user_id": "USR020"}
}

MERCHANT_TYPES = [
    "Grocery", "Electronics", "Restaurant", "Gas Station", "Online Shopping",
    "Travel", "Entertainment", "Healthcare", "Utilities", "Clothing"
]

CHANNELS = ["POS", "ATM", "Online"]


def authenticate_user(username: str, password: str) -> bool:
    if username in USERS:
        return USERS[username] == password
    return False


def get_user_info(username: str) -> dict:
    if username in USER_DATA:
        return USER_DATA[username].copy()
    return None


def get_user_id(username: str) -> str:
    if username in USER_DATA:
        return USER_DATA[username]["user_id"]
    return None


def ensure_directories():
    dirs = ["data/raw", "data/clean", "models", "docs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
