import numpy as np # linear algebra
import pandas as pd


t = pd.read_csv('datasets_188596_421248_Train-1542865627584.csv')
b = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
i = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
o = pd.read_csv('Train_Outpatientdata-1542865627584.csv')

tt = pd.read_csv('datasets_188596_421248_Test-1542969243754.csv')
tb = pd.read_csv('Test_Beneficiarydata-1542969243754.csv')
ti = pd.read_csv('Test_Inpatientdata-1542969243754.csv')
to = pd.read_csv('Test_Outpatientdata-1542969243754.csv')


(len(t), len(b), len(i), len(o))


df = pd.concat([i,o])
df = pd.merge(df, t, on="Provider", how="outer")
df = df.fillna(0)

df2 = pd.concat([ti,to])
df2 = pd.merge(df2, tt, on="Provider", how="outer")
df2 = df2.fillna(0)

# Labeling categorical data
from sklearn import preprocessing
catcols = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
        'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'AdmissionDt', 'ClmAdmitDiagnosisCode',
       'DischargeDt', 'DiagnosisGroupCode',
       'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6']

le = {}
X = pd.concat([df[catcols].astype(str), df2[catcols].astype(str)])
for i in catcols:
    print(i)
    le[i] = preprocessing.LabelEncoder()
    le[i].fit(X[i].astype(str))
    df[i] = le[i].transform(df[i].astype(str))
    df2[i] = le[i].transform(df2[i].astype(str))

# Preparing the dataset
cols = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'AdmissionDt', 'ClmAdmitDiagnosisCode',
       'DeductibleAmtPaid', 'DischargeDt', 'DiagnosisGroupCode',
       'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6']

X = df[cols]
Y = df["PotentialFraud"].apply(lambda x: True if x == "Yes" else False)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# Training
import xgboost as xgb
clf = xgb.XGBClassifier(n_jobs=12, n_estimators=200)
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["auc","error","logloss"],
        verbose=10)

