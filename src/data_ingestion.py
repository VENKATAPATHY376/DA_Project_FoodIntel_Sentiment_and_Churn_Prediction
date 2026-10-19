import os
import pandas as pd
from pathlib import Path
from src.utils import DATA_RAW

CANDIDATES = [
    "zomato-bangalore-restaurants.csv",
    "Zomato Bangalore Restaurants.csv",
    "zomato.csv",
    "food-delivery-dataset.csv",
    "online_food_dataset.csv",
    "train.csv",
    "test.csv",
]

def find_dataset():
    for fname in CANDIDATES:
        fpath = DATA_RAW / fname
        if fpath.exists():
            return fpath
    # any csv
    for f in DATA_RAW.glob("*.csv"):
        return f
    return None

def load_raw():
    data = {}
    found = find_dataset()
    if found is None:
        raise FileNotFoundError(f"No CSV in {DATA_RAW}")
    try:
        df = pd.read_csv(found)
        data["main"] = df
    except Exception as e:
        raise FileNotFoundError(f"Failed: {e}")
    return data, found

def merge_data(data):
    return data["main"].copy()
