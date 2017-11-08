import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


housing = pd.read_csv("cleanTrain.csv")
raw2 = pd.read_csv("cleanTrain.csv")
y = housing.SalePrice.values

def replace(df):
    isObject = (housing.dtypes == "object").values
    objectColumnNames = housing.columns[isObject]
    objectColumns = housing.select_dtypes(include=['object'])
    for colName in objectColumnNames:
        #print(housing[colName].value_counts())
        column = housing[colName].values
        cats = []
        #Collect unique list of categories
        for category in column:
            if category in cats:
                continue
            cats.append(category)
        #Replace category value with integer
        for index in range(len(column)):
            for category in cats:
                if category == column[index]:
                    column[index] = cats.index(category)
                if (type(category) == float) and (type(column[index]) == float) and (np.isnan(category) and np.isnan(column[index])):
                    column[index] = cats.index(category)
    housingObjects = housing.select_dtypes(include=[object])
    #encoder = OneHotEncoder()
    #encoder.fit(housingObjects)
    #print(encoder.feature_indices_)
    #oneHots = encoder.transform(housingObjects).toarray()
    #print(oneHots)
    #print(oneHots[1].reshape((1,-1)).shape)
    newHousing = (pd.get_dummies(housingObjects))
    newHousing['SalePrice'] = y
    return housing


def featurize(df):
    replace(df).to_csv("featurizedTrain.csv",index=False)
    newHousingCSV = pd.read_csv("featurizedTrain.csv")
    for column in newHousingCSV:
        print(newHousingCSV[column].value_counts())


featurize(housing)

#print(__name__)