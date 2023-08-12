import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.float_format", lambda x: '%.2f' % x)

####################################
#Sales Prediction with linear regresion
####################################

df = pd.read_csv("datasets/advertising.csv")

X = df[["TV"]]
y = df[["sales"]] 

#MODEL

reg_model = LinearRegression().fit(X,y)

#b
reg_model.intercept_[0]

#w
reg_model.coef_[0][0]

#TAHMİN

#150 BİRİMLİK TV HARCAMASINDA NE KADAR SATIŞ OLMASI BEKLENİR?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()
