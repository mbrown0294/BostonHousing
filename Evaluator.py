from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

housing = pd.read_csv("featurizedTrain.csv")
y = housing.SalePrice.values

def evaluate(df):
    X = housing.drop("SalePrice",1).values

    logReg = linear_model.LogisticRegression()
    logReg.fit(X, y)

    pricePred = logReg.predict(X)
    print(pricePred)
    priceTrue = y
    print(pricePred)
    print(mean_squared_error(priceTrue,pricePred))


#evaluate(housing,logReg) :

#evaluate(housing,linReg) : 867849985.001

#mean_squared_error(pricePred,priceTrue)

