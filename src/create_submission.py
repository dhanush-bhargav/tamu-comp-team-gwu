import data_crud

preds = data_crud.load_data("holdout", "predictions")
targets = data_crud.load_data("holdout", "target")

preds["id"] = preds["therapy_id"].str.split("-", n=1, expand=True).get(0)
preds["id"] = preds["id"].astype(int)
preds = preds[["id", "tgt_ade_dc_ind"]]

preds.set_index("id", inplace=True)
targets.set_index("id", inplace=True)

targets = targets.merge(preds, "left", left_index=True, right_index=True).fillna(preds["tgt_ade_dc_ind"].mean())

targets.reset_index(inplace=True)
targets = targets[["id", "tgt_ade_dc_ind"]]
data_crud.save_data(targets, "holdout", "submission")