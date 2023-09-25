import pandas as pd
from CONSTANTS import *


def load_data(type, name):
    if type == "train":
        return pd.read_csv(TRAINING_FOLDER + name + "_train.csv")
    if type == "holdout":
        return pd.read_csv(HOLDOUT_FOLDER + name + "_holdout.csv")

def save_data(df, type, name):
    if type == "train":
        df.to_csv(TRAINING_FOLDER + name + "_train.csv", index=False)
    if type == "holdout":
        df.to_csv(HOLDOUT_FOLDER + name + "_holdout.csv", index=False)