import pandas as pd
import numpy as np
from CONSTANTS import *
import data_crud

df = data_crud.load_data("holdout", "collated")

race_cd = pd.get_dummies(df["race_cd"])
race_cd.rename(columns={0.0: "race_cd_0", 1.0: "race_cd_1", 2.0: "race_cd_2", 3.0: "race_cd_3", 4.0: "race_cd_4", 5.0: "race_cd_5", 6.0: "race_cd_6"}, inplace=True)
df.drop(columns=["race_cd"], inplace=True)
df = df.join(race_cd)

# df.drop(columns=["seizure_diagnosis", "pain_diagnosis", "fatigue_diagnosis", "nausea_diagnosis", "hyperglycemia_diagnosis", "constipation_diagnosis", "diarrhea_diagnosis"], inplace=True)
# df["ade_treat_ind"] = df[["anticoag_ind", "diarrhea_treat_ind", "nausea_treat_ind", "seizure_treat_ind"]].sum(axis=1)
# df.drop(columns=["anticoag_ind", "diarrhea_treat_ind", "nausea_treat_ind", "seizure_treat_ind"], inplace=True)

df["est_age"] = (df["est_age"] - df["est_age"].mean()) / df["est_age"].std()

data_crud.save_data(df, "holdout", "collated_ready")