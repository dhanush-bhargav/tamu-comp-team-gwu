import data_crud
import pandas as pd
import numpy as np
from CONSTANTS import *
import pickle

def data_preprocess(type):

    target_df = data_crud.load_data(type, "target_clean")
    medclms_df = data_crud.load_data(type, "medclms_clean")
    rxclms_df = data_crud.load_data(type, "rxclms_clean")
    
    # target_df.fillna({"race_cd": 0, "est_age": int(target_df["est_age"].mean()), "cms_disabled_ind": 0, "cms_low_income_ind": 0, "sex_cd": "M"}, inplace=True)
    # target_df["sex_cd"].replace("M", 0, inplace=True)
    # target_df["sex_cd"].replace("F", 1, inplace=True)
    # data_crud.save_data(target_df, type, "target_clean")

    # medclms_df = medclms_df[["therapy_id", "visit_date", "ade_diagnosis", "seizure_diagnosis", "pain_diagnosis", "fatigue_diagnosis", "nausea_diagnosis",
    #                          "hyperglycemia_diagnosis", "constipation_diagnosis", "diarrhea_diagnosis"]]
    
    # print(medclms_df.head())
    # data_crud.save_data(medclms_df, type, "medclms_clean")

    # rxclms_df = rxclms_df[["therapy_id", "service_date", "pay_day_supply_cnt", "ddi_ind", "anticoag_ind", "diarrhea_treat_ind", "nausea_treat_ind", "seizure_treat_ind"]]
    # print(rxclms_df.head())
    # data_crud.save_data(rxclms_df, type, "rxclms_clean")

    # all_data = []

    # for name, group in target_df.groupby("therapy_id"):
    #     all_data.append((group, medclms_df[medclms_df["therapy_id"] == name], rxclms_df[rxclms_df["therapy_id"] == name]))
    
    target_df["therapy_start_date"] = pd.to_datetime(target_df["therapy_start_date"])
    target_df["therapy_end_date"] = pd.to_datetime(target_df["therapy_end_date"])
    medclms_df["visit_date"] = pd.to_datetime(medclms_df["visit_date"])
    rxclms_df["service_date"] = pd.to_datetime(rxclms_df["service_date"])

    medclms_df.sort_values("visit_date", inplace=True)
    rxclms_df.sort_values("service_date", inplace=True)
    target_df.drop(columns=["id"], inplace=True)
    target_df.set_index("therapy_id", inplace=True)

    medclms_df.set_index("visit_date", inplace=True)
    rxclms_df.set_index("service_date", inplace=True)

    all_data = []

    for name, group in target_df.groupby("therapy_id"):
        start_date = pd.to_datetime(group["therapy_start_date"].values[0] - pd.Timedelta(90, 'days'), utc=True)
        end_date = pd.to_datetime(group["therapy_end_date"].values[0], utc=True)
        
        medclms = medclms_df[(medclms_df["therapy_id"] == name)].resample('D').max(numeric_only=True)
        medclms = medclms[(medclms.index >= start_date)]
        medclms = medclms[(medclms.index <= end_date)]
        medclms = medclms.resample('M').sum()
       
        rxclms = rxclms_df[(rxclms_df["therapy_id"] == name)].resample('D').max(numeric_only=True)
        rxclms = rxclms[(rxclms.index >= start_date)]
        rxclms = rxclms[(rxclms.index <= end_date)]
        rxclms = rxclms.resample('M').sum()

        medclms = medclms.merge(rxclms, "outer", left_index=True, right_index=True).fillna(0)

        if len(medclms) > 0:
            clm_data = {"patient_data": group, "claim_data": medclms}
            all_data.append(clm_data)

    return all_data


if __name__ == "__main__":
    processed_data = data_preprocess("train")
    with open('all_data.pickle', 'wb') as f:
        pickle.dump(processed_data, f)