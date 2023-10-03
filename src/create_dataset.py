import torch
import numpy as np
import torch.utils.data as data_utils
import pandas as pd
import data_crud

df = data_crud.load_data("train", "collated_ready")
# df.drop(columns=["race_cd_0", "race_cd_1", "race_cd_2", "race_cd_3", "race_cd_4", "race_cd_5", "race_cd_6"], inplace=True)


labels = torch.tensor(df["tgt_ade_dc_ind"].values.astype(np.float32))
data = torch.tensor(df.drop(columns=["tgt_ade_dc_ind"]).values.astype(np.float32))
dataset = data_utils.TensorDataset(data, labels)

torch.save(dataset, "data/training/training_data.pt")
