import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from skimpy import skim

pd.set_option("display.float_format", lambda x: '%.2f' % x)

####################################
#Sales Prediction with linear regresion
####################################

df = pd.read_csv("datasets/advertising.csv")

# X = df[["TV"]]
# y = df[["sales"]] 

# #MODEL

# reg_model = LinearRegression().fit(X,y)

# #b
# reg_model.intercept_[0]

# #w
# reg_model.coef_[0][0]

# #TAHMİN

# #150 BİRİMLİK TV HARCAMASINDA NE KADAR SATIŞ OLMASI BEKLENİR?

# reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
#                 ci=False, color="r")

# g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
# g.set_ylabel("Satış Sayısı")
# g.set_xlabel("TV Harcamaları")
# # plt.xlim(-10, 310)
# # plt.ylim(bottom=0)
# # plt.show()


# #MSE 10.5
# y_pred = reg_model.predict(X)
# hata = mean_squared_error(y , y_pred)

# #RMSE 3.24
# hata2 = np.sqrt(hata)

# #MAE 2.54
# hata3 = mean_absolute_error(y , y_pred)

# print(hata,hata2,hata3)

# #R-KARE
# reg_model.score(X, y)

X = df.drop("sales", axis=1)
y = df[["sales"]]

#MODEL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

print(reg_model.coef_)

yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

# TAHMİN BAŞARISINI DEĞERLENDİRME

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))

#TRAİN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71
