import os
import random
from dotenv import load_dotenv

load_dotenv()

def get_random_api_key():
    keys = os.getenv("SCRAPIN_API_KEYS")
    key_list = keys.split(",")
    return random.choice(key_list)
