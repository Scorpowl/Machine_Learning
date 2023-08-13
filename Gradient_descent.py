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
    
def update_weight(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0,m):
        y_hat = b + w * x[i]
        y_ger = Y[i]

        b_deriv_sum += (y_hat - y_ger)
        w_deriv_sum += (y_hat - y_ger) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weight(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter = {:d}    b = {:.2f}  mse= {:.4f}".format(i,b,w,mse))
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters,b,w, cost_function(Y,b,w,X)))
    return cost_history, b ,w 

    





