import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from skimpy import skim

pd.set_option("display.float_format", lambda x: '%.2f' % x)

df = pd.read_csv("datasets/advertising.csv")

#MSE FUNCTION
#Y = bağımlı değişken sayısı
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(m):
        y_tah = b + w*X[i]
        y_ger = Y[i]
        sse += (y_tah - y_ger) **2 
        mse = sse / m
        return mse
    





