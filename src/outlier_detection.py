import numpy as np
import data_crud
import xgboost
from sklearn.metrics import roc_auc_score


df = data_crud.load_data("train", "collated_ready")

train = df.sample(frac=0.9, random_state=100)
test = df.drop(index=train.index)

train_labels = train["tgt_ade_dc_ind"].to_numpy()
test_labels = test["tgt_ade_dc_ind"].to_numpy()

train = train.drop(columns=["tgt_ade_dc_ind"])
test = test.drop(columns=["tgt_ade_dc_ind"])


# model = xgboost.XGBClassifier(max_depth=10, n_estimators=100, objective="binary:logistic")
# model.fit(train, train_labels)

# print(roc_auc_score(test_labels, model.predict_proba(test).T[1]))

# model.save_model("xgboost_model.json")