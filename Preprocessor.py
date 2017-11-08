import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = pd.read_csv("train.csv")
raw1 = pd.read_csv("train.csv")
y = housing.SalePrice.values

def numberClean():
    housing.drop("SalePrice",1,inplace=True)
    isNumeric = ((housing.dtypes == "int64") | (housing.dtypes == "float64")).values
    numericColumns = housing.columns[isNumeric]
    colCount = numericColumns.shape[0]
    for colName in numericColumns:
        column = housing[colName].values
        median = np.nanmedian(column)
        housing.at[np.isnan(column),colName] = median
    #Sums number of NaN values:
        #print(np.sum(np.isnan(isNumeric)))
    #print(numericColumns)
    #print(objectColumns)


def objectClean():
    isObject = (housing.dtypes == "object").values
    objectColumns = housing.columns[(isObject)]
    colCount = objectColumns.shape[0]
    X = housing[objectColumns].values
    housing['CentralAir'] = np.where(housing.CentralAir=='Y',1,0)
    #print(housing.PavedDrive)
    housing.PavedDrive = np.where(housing.PavedDrive=='Y',1,0)
    #print(housing.PavedDrive)


def clean():
    objectClean()
    numberClean()
    housing['SalePrice'] = y
    housing.to_csv("cleanTrain.csv", index=False)
    newHousing = pd.read_csv("cleanTrain.csv")



clean()
#print(raw)
#print(housing.PavedDrive.value_counts())
#objectClean()
#numberClean()