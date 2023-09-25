import data_crud
import pandas as pd
import numpy as np
from CONSTANTS import *
import pickle

with open("all_data.pickle", "rb") as f:
    data = pickle.load(f)


