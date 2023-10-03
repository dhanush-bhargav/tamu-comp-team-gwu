import numpy as np
import data_crud
import xgboost

model = xgboost.XGBClassifier()
model.load_model("xgboost_model.json")

df = data_crud.load_data("holdout", "collated_ready")
df["race_cd_6"] = 0

X = df.drop(columns=["therapy_id"]).to_numpy()
y = model.predict_proba(X).T[1]

df["tgt_ade_dc_ind"] = y

data_crud.save_data(df, "holdout", "predictions")