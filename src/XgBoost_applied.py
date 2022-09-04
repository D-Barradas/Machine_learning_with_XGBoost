import os
import pandas as pd
import numpy as np
import pickle
import sklearn as skl
import xgboost

np.random.seed(101)

print(f"Pandas  Version: {pd.__version__}")
print(f"Sklearn Version: {skl.__version__}")
print(f"Pickle Version : {pickle.format_version}")
print(f"Xgboost Version: {xgboost.__version__}")

def load(file_name):
    b = {}
    try:
        b = pickle.load(open(file_name, "rb"))
        print("Loading Successful")
        return b
    except (OSError, IOError) as e:
        print("Loading Failed. Initializing to empty")
        return e


def load_data(file_name):
    features = ["AP_DFIRE2", "AP_PISA" ,"AP_T1", "AP_T2", "CP_MJ3h" ,"SIPPER", "ELE", "VDW", "PYDOCK_TOT", "AP_dDFIRE"]
    print (f"Loading data from {file_name}")
    folder = os.listdir(file_name)

    my_dataframes = []
    for f in folder:
        print (f"Loading data from {f}")
        df_temp = pd.read_csv(f"{file_name}/{f}",index_col="Conf") 
        df_temp["BM_ID"] = f[0:4]
        my_dataframes.append(df_temp)
    data_LD = pd.concat(my_dataframes)
    return data_LD[features],data_LD["BM_ID"] 

# create a funtion to get the current working directory one level up
def getcwd_up():
    cwd = os.getcwd()
    #print(f"Current working directory: {cwd}")
    return cwd


mydir = getcwd_up()
ld_pd,df_id_list = load_data(f"{mydir}/data/results_LD_v3/")
index=ld_pd.index


classifer = f"{mydir}/models/Xgboost_LD_hp3_LD.sav"
my_cls = load(classifer)
scaler = f"{mydir}/models/scaler_LD_BM4_all_features.sav"
scaler = load(scaler)

ld_pd = scaler.transform(ld_pd)
predictions = my_cls.predict(ld_pd)
probability = my_cls.predict_proba(ld_pd)

predictions = pd.DataFrame(data=predictions,index=index)
probability = pd.DataFrame(data=probability,index=index)

predictions.columns = ["Prediction"]
probability.columns = ["Proba_incorrect","Proba_correct"]

results = pd.concat([predictions,probability,df_id_list],axis=1)
#.to_csv("../data/results_LD_v2/predictions_Xgboost_LD_hp3_LD.csv")
results.to_csv(f"{mydir}/results/predictions_Xgboost_LD_hp3_LD_v3.csv")
print ( results.sort_values(by=["BM_ID","Proba_correct"],ascending=False).head(n=10) ) 