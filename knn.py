import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

df = pd.read_csv("datasets/diabetes.csv")
# print(df)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)
# print(X_scaled)

X = pd.DataFrame(X_scaled, columns=X.columns)

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1,random_state=45)
# print(random_user)

knn_model.predict(random_user)

# Confusion matrix için y_pred: 
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:,1]

# print(classification_report(y , y_pred))

AUC = roc_auc_score(y, y_prob)

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# print(cv_results)

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# # 0.73
# # 0.59
# # 0.78

# # 1. Örnek boyutu arttıralabilir.
# # 2. Veri ön işleme
# # 3. Özellik mühendisliği
# # 4. İlgili algoritma için optimizasyonlar yapılabilir.

# ################################################
# # 5. Hyperparameter Optimization
# ################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2,50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_
#17

# ################################################
# # 6. Final Model
# ################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)