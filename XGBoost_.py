import pyecharts.options as opts
from pyecharts.charts import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Load data
data = pd.read_csv('SMOTENC_data.csv')
X = data[['Aerobic plate counts', 'Escherichia coli', "Salmonella", "Listeria monocytogenes", "Bacillus cereus",
          'Economic level', 'Month', 'Classification', "Year"]]
Y = data['State']

# Data partitioning
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=333)

# Training XGBoost model
model = xgb.XGBClassifier(learning_rate=0.2, random_state=5)
model = model.fit(X_train, Y_train.astype('int'))
result = model.score(X_test, Y_test.astype('int'))
predictions = model.predict(X_test)
resultquan = model.predict(X)
y_pro = model.predict_proba(X_test)

# Check the accuracy of testing
i = 'XGboost'
print(f'{i}  Accuracy:', accuracy_score(Y_test, predictions))
print(f'{i}  Precison:', precision_score(Y_test, predictions))
print(f'{i}  Recall  :', recall_score(Y_test, predictions))
print(f'{i}  F1      :', f1_score(Y_test, predictions))
print(f'{i}  Auc', roc_auc_score(Y_test, y_pro[:, 1]))

# The weight obtained by XGBoost
plt.figure(figsize=(16, 6))
(pd.Series(model.feature_importances_, index=X.columns)
   .nlargest(len(X.columns))
   .plot(kind='barh'))
# plt.show()

# Weight normalization
total = []
for i in range(5):
    total.append(model.feature_importances_[i])
total_importances = sum(total)

APC = model.feature_importances_[0] / total_importances
Ecoli = model.feature_importances_[1] / total_importances
Salmonella = model.feature_importances_[2] / total_importances
Lmonocytogenes = model.feature_importances_[3] / total_importances
Bcereus = model.feature_importances_[4] / total_importances

# Expert rating adjusts weight
ExpertAPC = APC*3
ExpertEcoli = Ecoli*4
ExpertSalmonella = Salmonella*3
ExpertLmonocytogenes = Lmonocytogenes*4
ExpertBcereus = Bcereus*2
totalexpert_importances = ExpertAPC + ExpertEcoli + ExpertSalmonella + ExpertLmonocytogenes + ExpertBcereus

# Expert rating adjustment weight normalization
FinalAPC = ExpertAPC/totalexpert_importances
FinalEcoli = ExpertEcoli/totalexpert_importances
FinalSalmonella1 = ExpertSalmonella/totalexpert_importances
FinalLmonocytogenes = ExpertLmonocytogenes/totalexpert_importances
FinalBcereus = ExpertBcereus/totalexpert_importances

# Calculate the quality and safety score of students' meals
# data = pd.read_excel('Studentmeal_data.xls')
#
# x1 = np.array([0, 10000])
# y1 = np.array([100, 80])
# model1 = LinearRegression().fit(x1.reshape(-1, 1), y1.reshape(-1, 1))
#
# x2 = np.array([10000, 100000])
# y2 = np.array([80, 60])
# model2 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# x3 = np.array([100000, 200000])
# y3 = np.array([60, 0])
# model3 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# APC_Score = []
# for h in range(len(data['Aerobic plate counts'])):
#     i = np.array(data['Aerobic plate counts'][h]).reshape(1, -1)
#     if i <= 10000:
#         APC_Score.append(model1.predict(i)[0][0])
#     if i> 10000 and i <= 100000:
#         APC_Score.append(model2.predict(i)[0][0])
#     if i> 100000 and i <= 200000:
#         APC_Score.append(model3.predict(i)[0][0])
#     if i > 200000:
#         APC_Score.append(0)
#
# Salmonella_Score = []
# for h in range(len(data['Salmonella'])):
#     if data['Salmonella'][h] == 0:
#         Salmonella_Score.append(100)
#     else:
#         Salmonella_Score.append(0)
#
# Lmonocytogenes_Score = []
# for h in range(len(data['Listeria monocytogenes'])):
#     if data['Listeria monocytogenes'][h] == 0:
#         Lmonocytogenes_Score.append(100)
#     else:
#         Lmonocytogenes_Score.append(0)
#
# x1 = np.array([0, 20])
# y1 = np.array([100, 80])
# model1 = LinearRegression().fit(x1.reshape(-1, 1), y1.reshape(-1, 1))
#
# x2 = np.array([20, 100])
# y2 = np.array([80, 60])
# model2 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# x3 = np.array([100, 200])
# y3 = np.array([60, 0])
# model3 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# Ecoli_Score = []
# for h in range(len(data['Escherichia coli'])):
#     i = np.array(data['Escherichia coli'][h]).reshape(1, -1)
#     if i <= 20:
#         Ecoli_Score.append(model1.predict(i)[0][0])
#     if i > 20 and i <= 100:
#         Ecoli_Score.append(model2.predict(i)[0][0])
#     if i > 100 and i <= 200:
#         Ecoli_Score.append(model3.predict(i)[0][0])
#     if i > 200:
#         Ecoli_Score.append(0)
#
# x1 = np.array([0, 1000])
# y1 = np.array([100, 80])
# model1 = LinearRegression().fit(x1.reshape(-1, 1), y1.reshape(-1, 1))
#
# x2 = np.array([1000, 100000])
# y2 = np.array([80, 60])
# model2 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# x3 = np.array([100000, 200000])
# y3 = np.array([60, 0])
# model3 = LinearRegression().fit(x2.reshape(-1, 1), y2.reshape(-1, 1))
#
# Bcereus_Score = []
# for h in range(len(data['Bacillus cereus'])):
#     i = np.array(data['Bacillus cereus'][h]).reshape(1, -1)
#     if i <= 1000:
#         Bcereus_Score.append(model1.predict(i)[0][0])
#     if i> 1000 and i <= 100000:
#         Bcereus_Score.append(model2.predict(i)[0][0])
#     if i> 100000 and i <= 200000:
#         Bcereus_Score.append(model3.predict(i)[0][0])
#     if i > 200000:
#         Bcereus_Score.append(0)
#
# data['APC_Score'] = APC_Score
# data['Ecoli_Score'] = Ecoli_Score
# data['Bcereus_Score'] = Bcereus_Score
# data['Salmonella_Score'] = Salmonella_Score
# data['Lmonocytogene_Score'] = Lmonocytogenes_Score
#
# total_Score = []
# for h in range(len(data['State'])):
#     if data['State'][h] == 1:
#         total_Score.append(0.6*(FinalAPC*data['APC_Score'][h] + FinalEcoli*data['Ecoli_Score'][h]
#                              + FinalBcereus*data['Bcereus_Score'][h]
#                              + FinalSalmonella1*data['Salmonella_Score'][h] + FinalLmonocytogenes*data['Lmonocytogene_Score'][h]))
#     else:
#         total_Score.append(FinalAPC * data['APC_Score'][h] + FinalEcoli * data['Ecoli_Score'][h]
#                                + FinalBcereus * data['Bcereus_Score'][h]
#                                + FinalSalmonella1 * data['Salmonella_Score'][h] + FinalLmonocytogenes*data['Lmonocytogene_Score'][h])
#
# data['total_Score'] = total_Score
# data.to_excel('Score_table.xls')