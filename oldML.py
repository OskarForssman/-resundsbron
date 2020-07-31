# For loading the data
import pandas as pd
import datetime
import numpy as np

# For visualization
import matplotlib.pyplot as plt

# For Modeling
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
import pandas as pd
import datetime
import numpy as np
import os as os
# For visualization
import matplotlib.pyplot as plt
import numpy as np
# For Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error,\
                accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
from math import *
import pdb

# from sklearn import svm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# Specify classifcation/regression
ml_type = 'classification' # classifcation/regression

# To display data in full

df_dataset = pd.read_csv('osb_pdm_dataset.csv', sep=';', error_bad_lines=False)

df_dataset.loc[:, 'Asset Failure'] = 0
#real errors
df_dataset.iloc[1154  , df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[16040, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[65023, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[90668, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[91364, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[91567, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[94726, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[105890, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[106533, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[107379, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[120865, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[121474, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[122354, df_dataset.columns.get_loc('Asset Failure')] = 1
df_dataset.iloc[123859, df_dataset.columns.get_loc('Asset Failure')] = 1



#Noise to much decimals in dataset

df_dataset=df_dataset.round(2)







#Replace Nan and null values
df_dataset['HuPV Mean 14D'].fillna(df_dataset['HuPV Mean 14D'].mean(), inplace=True)
df_dataset['HuSP Mean 14D'].fillna(df_dataset['HuSP Mean 14D'].mean(), inplace=True)
df_dataset['Pres Mean 14D'].fillna(df_dataset['Pres Mean 14D'].mean(), inplace=True)
df_dataset['Temp Mean 14D'].fillna(df_dataset['Temp Mean 14D'].mean(), inplace=True)
df_dataset['HuPV Std 14D'].fillna(df_dataset['HuPV Std 14D'].mean(), inplace=True)
df_dataset['HuSP Std 14D'].fillna(df_dataset['HuSP Std 14D'].mean(), inplace=True)
df_dataset['Pres Std 14D'].fillna(df_dataset['Pres Std 14D'].mean(), inplace=True)
df_dataset['Temp Std 14D'].fillna(df_dataset['Temp Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 AirTemp Mean 14D'].fillna(df_dataset['54WES0001 AirTemp Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 AirTemp Mean 14D'].fillna(df_dataset['54WES0002 AirTemp Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 AirTemp Mean 14D'].fillna(df_dataset['54WES0003 AirTemp Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 AirTemp Mean 14D'].fillna(df_dataset['54WES0004 AirTemp Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 AirTemp Mean 14D'].fillna(df_dataset['54WES0005 AirTemp Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 Humidity Mean 14D'].fillna(df_dataset['54WES0001 Humidity Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 Humidity Mean 14D'].fillna(df_dataset['54WES0002 Humidity Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 Humidity Mean 14D'].fillna(df_dataset['54WES0003 Humidity Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 Humidity Mean 14D'].fillna(df_dataset['54WES0004 Humidity Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 Humidity Mean 14D'].fillna(df_dataset['54WES0005 Humidity Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindDir Mean 14D'].fillna(df_dataset['54WES0001 WindDir Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindDir10 Mean 14D'].fillna(df_dataset['54WES0001 WindDir10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindDir Mean 14D'].fillna(df_dataset['54WES0002 WindDir Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindDir10 Mean 14D'].fillna(df_dataset['54WES0002 WindDir10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindDir Mean 14D'].fillna(df_dataset['54WES0003 WindDir Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindDir10 Mean 14D'].fillna(df_dataset['54WES0003 WindDir10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindDir Mean 14D'].fillna(df_dataset['54WES0004 WindDir Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindDir10 Mean 14D'].fillna(df_dataset['54WES0004 WindDir10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindDir Mean 14D'].fillna(df_dataset['54WES0005 WindDir Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindDir10 Mean 14D'].fillna(df_dataset['54WES0005 WindDir10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed Mean 14D'].fillna(df_dataset['54WES0001 WindSpeed Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed10 Mean 14D'].fillna(df_dataset['54WES0001 WindSpeed10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed30 Mean 14D'].fillna(df_dataset['54WES0001 WindSpeed30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeedEffective Mean 14D'].fillna(df_dataset['54WES0001 WindSpeedEffective Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeedMax30 Mean 14D'].fillna(df_dataset['54WES0001 WindSpeedMax30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed Mean 14D'].fillna(df_dataset['54WES0002 WindSpeed Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed10 Mean 14D'].fillna(df_dataset['54WES0002 WindSpeed10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed30 Mean 14D'].fillna(df_dataset['54WES0002 WindSpeed30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeedEffective Mean 14D'].fillna(df_dataset['54WES0002 WindSpeedEffective Mean 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeedMax30 Mean 14D'].fillna(df_dataset['54WES0002 WindSpeedMax30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed Mean 14D'].fillna(df_dataset['54WES0003 WindSpeed Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed10 Mean 14D'].fillna(df_dataset['54WES0003 WindSpeed10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed30 Mean 14D'].fillna(df_dataset['54WES0003 WindSpeed30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeedEffective Mean 14D'].fillna(df_dataset['54WES0003 WindSpeedEffective Mean 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeedMax30 Mean 14D'].fillna(df_dataset['54WES0003 WindSpeedMax30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed Mean 14D'].fillna(df_dataset['54WES0004 WindSpeed Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed10 Mean 14D'].fillna(df_dataset['54WES0004 WindSpeed10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed30 Mean 14D'].fillna(df_dataset['54WES0004 WindSpeed30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeedEffective Mean 14D'].fillna(df_dataset['54WES0004 WindSpeedEffective Mean 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeedMax30 Mean 14D'].fillna(df_dataset['54WES0004 WindSpeedMax30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed Mean 14D'].fillna(df_dataset['54WES0005 WindSpeed Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed10 Mean 14D'].fillna(df_dataset['54WES0005 WindSpeed10 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed30 Mean 14D'].fillna(df_dataset['54WES0005 WindSpeed30 Mean 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeedEffective Mean 14D'].fillna(df_dataset['54WES0005 WindSpeedEffective Mean 14D'].mean(), inplace=True)
df_dataset['54WES0001 AirTemp Std 14D'].fillna(df_dataset['54WES0001 AirTemp Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 AirTemp Std 14D'].fillna(df_dataset['54WES0002 AirTemp Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 AirTemp Std 14D'].fillna(df_dataset['54WES0003 AirTemp Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 AirTemp Std 14D'].fillna(df_dataset['54WES0004 AirTemp Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 AirTemp Std 14D'].fillna(df_dataset['54WES0005 AirTemp Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 Humidity Std 14D'].fillna(df_dataset['54WES0001 Humidity Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 Humidity Std 14D'].fillna(df_dataset['54WES0002 Humidity Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 Humidity Std 14D'].fillna(df_dataset['54WES0003 Humidity Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 Humidity Std 14D'].fillna(df_dataset['54WES0004 Humidity Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 Humidity Std 14D'].fillna(df_dataset['54WES0005 Humidity Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindDir Std 14D'].fillna(df_dataset['54WES0001 WindDir Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindDir10 Std 14D'].fillna(df_dataset['54WES0001 WindDir10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindDir Std 14D'].fillna(df_dataset['54WES0002 WindDir Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindDir10 Std 14D'].fillna(df_dataset['54WES0002 WindDir10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindDir Std 14D'].fillna(df_dataset['54WES0003 WindDir Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindDir10 Std 14D'].fillna(df_dataset['54WES0003 WindDir10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindDir Std 14D'].fillna(df_dataset['54WES0004 WindDir Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindDir10 Std 14D'].fillna(df_dataset['54WES0004 WindDir10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindDir Std 14D'].fillna(df_dataset['54WES0005 WindDir Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindDir10 Std 14D'].fillna(df_dataset['54WES0005 WindDir10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed Std 14D'].fillna(df_dataset['54WES0001 WindSpeed Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed10 Std 14D'].fillna(df_dataset['54WES0001 WindSpeed10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeed30 Std 14D'].fillna(df_dataset['54WES0001 WindSpeed30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeedEffective Std 14D'].fillna(df_dataset['54WES0001 WindSpeedEffective Std 14D'].mean(), inplace=True)
df_dataset['54WES0001 WindSpeedMax30 Std 14D'].fillna(df_dataset['54WES0001 WindSpeedMax30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed Std 14D'].fillna(df_dataset['54WES0002 WindSpeed Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed10 Std 14D'].fillna(df_dataset['54WES0002 WindSpeed10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeed30 Std 14D'].fillna(df_dataset['54WES0002 WindSpeed30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeedEffective Std 14D'].fillna(df_dataset['54WES0002 WindSpeedEffective Std 14D'].mean(), inplace=True)
df_dataset['54WES0002 WindSpeedMax30 Std 14D'].fillna(df_dataset['54WES0002 WindSpeedMax30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed Std 14D'].fillna(df_dataset['54WES0003 WindSpeed Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed10 Std 14D'].fillna(df_dataset['54WES0003 WindSpeed10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeed30 Std 14D'].fillna(df_dataset['54WES0003 WindSpeed30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeedEffective Std 14D'].fillna(df_dataset['54WES0003 WindSpeedEffective Std 14D'].mean(), inplace=True)
df_dataset['54WES0003 WindSpeedMax30 Std 14D'].fillna(df_dataset['54WES0003 WindSpeedMax30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed Std 14D'].fillna(df_dataset['54WES0004 WindSpeed Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed10 Std 14D'].fillna(df_dataset['54WES0004 WindSpeed10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeed30 Std 14D'].fillna(df_dataset['54WES0004 WindSpeed30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeedEffective Std 14D'].fillna(df_dataset['54WES0004 WindSpeedEffective Std 14D'].mean(), inplace=True)
df_dataset['54WES0004 WindSpeedMax30 Std 14D'].fillna(df_dataset['54WES0004 WindSpeedMax30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed Std 14D'].fillna(df_dataset['54WES0005 WindSpeed Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed10 Std 14D'].fillna(df_dataset['54WES0005 WindSpeed10 Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeed30 Std 14D'].fillna(df_dataset['54WES0005 WindSpeed30 Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeedEffective Std 14D'].fillna(df_dataset['54WES0005 WindSpeedEffective Std 14D'].mean(), inplace=True)
df_dataset['54WES0005 WindSpeedMax30 Std 14D'].fillna(df_dataset['54WES0005 WindSpeedMax30 Std 14D'].mean(), inplace=True)

#Split Time into lesser parts!
splitTime = df_dataset["Time"].str.split(" ", n=1, expand=True)
splitDate = splitTime.get(0).str.split("-", n=2, expand=True)
SplitClock = splitTime.get(1).str.split(":", n=2, expand=True)
year = splitDate.get(0)
month = splitDate.get(1)
day = splitDate.get(2)
ClockTime = SplitClock.get(0)
df_dataset['year'] = year
df_dataset['month'] = month
df_dataset['day'] = day
df_dataset['ClockTime'] = ClockTime

df_dataset['year'] = df_dataset.year.apply(float)
df_dataset['month'] = df_dataset.month.apply(float)
df_dataset['day'] = df_dataset.day.apply(float)
df_dataset['ClockTime'] = df_dataset.ClockTime.apply(float)

splitassetNumb=df_dataset["AssetNumber"].str.split("DHS",n=1,expand=True)
df_dataset['AssetNumb']=splitassetNumb.get(1)

real = ['Asset Failure', 'year', 'month', 'day', 'ClockTime', 'AssetNumb']
remove_features = ['AssetNumb','Asset Failure', 'Time', 'AssetNumber', 'Unnamed: 0', 'month', 'day', 'ClockTime', 'year']
y = df_dataset[real]

X = df_dataset.drop(remove_features, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

# RF Classifier

if ml_type == 'classification':
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train.astype(np.float32))

    
     #kNeigh=KNeighborsClassifier(3)
     #kNeigh.fit(X_train,y_train.astype(np.float32))

if ml_type == 'classification':
    preds_train = pd.DataFrame(rf.predict(X_train), columns=y_train.columns)
    preds_test = pd.DataFrame(rf.predict(X_test), columns=y_train.columns)
    print(y_train.astype(np.float32).iloc[:, 0])
    print(preds_train.iloc[:, 0])
    train_f1 = f1_score(y_train.astype(np.float32).iloc[:, 0], preds_train.iloc[:, 0], average='weighted')
   
    train_precision = precision_score(y_train.astype(np.float32).iloc[:,0], preds_train.iloc[:,0],average='weighted')
    train_recall = recall_score(y_train.astype(np.float32).iloc[:, 0], preds_train.iloc[:,0],average='weighted')
    test_f1 = f1_score(y_test.astype(np.float32).iloc[:,0], preds_test.iloc[:,0],average='weighted'),
    test_precision = precision_score(y_test.astype(np.float32).iloc[:,0], preds_test.iloc[:,0],average='weighted')
    test_recall = recall_score(y_test.astype(np.float32).iloc[:, 0], preds_test.iloc[:, 0],average='weighted')
    print('rf TRAIN: f1_score={}, precision={}, recall={}'.format(train_f1, train_precision, train_recall))
    print('rf TEST: f1_score={}, precision={}, recall={}'.format(test_f1, test_precision, test_recall))

