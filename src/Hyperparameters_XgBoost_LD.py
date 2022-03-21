#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report,matthews_corrcoef
import sklearn as skl
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing
from sklearn.model_selection import KFold
np.random.seed(101)

#%%
PDB_BM5 = [
'1EXB','1JTD','1M27','1RKE','2A1A','2GAF','2GTP','2VXT','2W9E',
'2X9A','2YVJ','3A4S','3AAA','BAAD','3AAD','3BIW','3BX7',
'3DAW','3EO1','3EOA','3F1P','3FN1','3G6D','3H11',
'3H2V','3HI6','3HMX','3K75','3L5W','3L89','3LVK','3MXW',
'BP57','CP57','3P57','3PC8','3R9A','3RVW','3S9D','3SZK',
'3V6Z','3VLB','4DN4','4FQI','4FZA','4G6J','4G6M','4GAM',
'4GXU','4H03','4HX3','4IZ7','4JCV','4LW4','4M76'
]

print(f"Pandas  Version: {pd.__version__}")
print(f"Sklearn Version: {skl.__version__}")
print(f"Pickle Version : {pickle.format_version}")
print(f"Xgboost Version: {xgboost.__version__}")

##functions to store and load classifiers thresholds (cache values to save recalculation time)
def store(b, file_name):
    pickle.dump(b, open(file_name, "wb"))

def load(file_name):
    b = {}
    try:
        b = pickle.load(open(file_name, "rb"))
        print("Loading Successful")
        return b
    except (OSError, IOError) as e:
        print("Loading Failed. Initializing to empty")
        b = {}
        return b


def load_data_sets():
    features = ["AP_DFIRE2", "AP_PISA" ,"AP_T1", "AP_T2", "CP_MJ3h" ,"SIPPER", "ELE", "VDW", "PYDOCK_TOT", "AP_dDFIRE"]

    all_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_all_data_ccharppi_4_march_2020_complete.csv",dtype={'class_q': 'object'})
    all_balanced_data.set_index('Conf',inplace=True)
    all_balanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()
    print (all_balanced_data.shape)
    all_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_all_data_ccharppi_4_march_2020_complete.csv",dtype={'class_q': 'object'})
    all_unbalanced_data.set_index('Conf',inplace=True)
    all_unbalanced_data.loc["Z_1JTG_1136_M.pdb","DDG_V"]  = all_balanced_data["DDG_V"].mean()


#     Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set_feb_12_2021.csv")
    Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set.csv")
#     Scorers_balanced_data = pd.read_csv("../data/Clean_dataframe_balanced_scorers_set_march_22_2021.csv")
    
    print (Scorers_balanced_data.shape)


    Scorers_balanced_data.set_index('Conf',inplace=True)
    Scorers_balanced_data.dropna(inplace=True)

#     Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set_feb_12_2021.csv")
    Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set.csv")
#     Scorers_unbalanced_data = pd.read_csv("../data/Clean_dataframe_unbalanced_scorers_set_march_22_2021.csv")


    Scorers_unbalanced_data.set_index('Conf',inplace=True)
    Scorers_unbalanced_data.dropna(inplace=True)

    X_train = all_balanced_data[~all_balanced_data["idx"].isin(PDB_BM5) ][features]
    y_train = all_balanced_data[~all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')

            ## data set for less than 5 
    X_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ][features]
    y_val = all_balanced_data[all_balanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')
    #         print (X_test.size,y_test.size)
            ## data set for less than 5 
    X_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ][features]
    y_val_u = all_unbalanced_data[all_unbalanced_data["idx"].isin(PDB_BM5) ]["label_binary"].astype('bool')
    
    X_test = Scorers_balanced_data[features]
    y_test = Scorers_balanced_data["binary_label"].astype('bool')
    
    X_test_u = Scorers_unbalanced_data[features]
    y_test_u = Scorers_unbalanced_data["binary_label"].astype('bool')
    
    X_test_u.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                'binary_label':'label_binary'
                            },inplace=True)
    X_test.rename(columns={'NIS Polar' :'Nis_Polar',
                                  'Nis Apolar':'Nis_Apolar',
                                  'BSA Apolar':'BSA_Apolar',
                                  'BSA Polar' :'BSA_Polar',
                                   'binary_label':'label_binary'
                          },inplace=True)
    
#     for x in X_val_u.columns:
#         if x not in X_test.columns:
#             print (x)
    return X_train, y_train , X_val, y_val, X_test, y_test ,X_val_u, y_val_u, X_test_u, y_test_u 

def scaling_data(X_train,X_test,X_test_unbalanced):
    """ As the name implies this definition scale the data
    Ideally the imputs shupold be pandas dataframes 

    Args:
        X_train (dataframe): Balanced training data
        X_test (dataframe): Validation/test BALANCED data to scale according to the fit of the training
        X_test_unbalanced ([type]): Validation/test UNBALANCED data to scale

    Returns:
        scaled_train,scaled_test,scaled_test_u : scaled data
    """
#     scaler = MinMaxScaler()
    features = ['idx','class_q','pdb1','chains_pdb1','pdb2','chains_pdb2',
                'label_binary','DQ_val','binary_label','identification','labels']
    scaler = StandardScaler()

    for x in features :
        if x in X_train.columns:
            X_train= X_train.drop(x,axis=1)            
    for x in features :
        if x in X_test.columns: 
                X_test = X_test.drop(x,axis=1)
            
    for x in features :
        if x in X_test_unbalanced.columns:
            X_test_unbalanced= X_test_unbalanced.drop(x,axis=1)

    scaler.fit(X_train)
    
    filename = '../models/scaler_LD_BM4_all_features.sav'
    pickle.dump(scaler, open(filename, 'wb'))
    
    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)
    scaled_test_u = scaler.transform(X_test_unbalanced)
    return scaled_train,scaled_test,scaled_test_u

def save_metrics_results(model,x_test,y_test,tag):
    # target_names = ['Incorrect', 'Correct']

    y_pred = model.predict(x_test)
    cr = classification_report(y_true=y_test, y_pred=y_pred,output_dict=True,zero_division=0)
    mmc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    # print (cr)
    acc = cr["accuracy"]
    rec_false = cr["False"]["recall"]
    rec_true  = cr["True"]["recall"]
    pres_false = cr["False"]["precision"]
    pres_true = cr["True"]["precision"]
    f1_false =  cr["False"]["f1-score"]
    f1_true =  cr["True"]["f1-score"]

    results = {
        "Acc": acc,
        "R_inc":rec_false,
        "R_cor":rec_true,
        "P_inc":pres_false ,
        "P_cor":pres_true,
        "F1_inc":f1_false,
        "F1_cor":f1_true,
        "MCC":mmc
    }
    mean_df = pd.DataFrame(data=results,index=[f"{tag}"])
    return mean_df
#%%
my_seed = np.random.seed(101) 
xbg_class = XGBClassifier(use_label_encoder=False,n_jobs=multiprocessing.cpu_count() ,random_state=my_seed,eval_metric='logloss')

#%%
random_grid = {
'max_depth': [2, 4, 6],
'n_estimators': [50, 100, 200],
'min_child_weight': [1, 3, 5],
'gamma': [0.5, 1, 1.5, 2, 5],
'colsample_bytree': [0.3, 0.5, 0.7],
'colsample_bylevel': [0.3, 0.5, 0.7],
'learning_rate': [0.01, 0.1, 0.2],
'objective': ['binary:logistic','binary:logitraw','binary:hinge'],
'eval_metric': ['logloss','auc','error'],
'booster':  ['gbtree','gblinear','dart'],
'base_score': [0.5],
'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'rate_drop': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'normalize_type': ['tree','forest']
}
#%%
### load the data ###
X_train, y_train , X_val, y_val, X_test, y_test ,X_val_u, y_val_u, X_test_u, y_test_u  = load_data_sets()

#%%
### scale data ### 
# X_train_w, X_test_w, X_test_u_w = scaling_data(X_train, X_test, X_test_u)
X_train_w, X_val_w, X_val_u_w = scaling_data(X_train, X_val, X_val_u)
print (X_train_w.shape)
#%%
cv = KFold(10, shuffle=True)
#%%
clf = RandomizedSearchCV(estimator = xbg_class,n_iter=100,param_distributions = random_grid, verbose=2, cv=cv,n_jobs=-1, random_state=my_seed)
clf.fit(X_train_w, y_train)
print(clf.best_score_)
print(clf.best_params_)

# %%
store(b=clf.best_estimator_,file_name=f"../models/Xgboost_LD_hp5_LD.sav")