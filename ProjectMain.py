import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

x1 = np.array([])
x2 = np.array([])
x3 = np.array([])
x4 = np.array([])
x5 = np.array([])
x6 = np.array([])
y = np.array([])
x1a = np.array([])
x2a = np.array([])
x3a = np.array([])
x4a = np.array([])
x5a = np.array([])
x6a = np.array([])

with open("CE 475 Fall 2019 Project Data - Data.csv", encoding="Latin-1") as f:
    csv_list = list(csv.reader(f))

firstHundred = csv_list[:101]
lastTwenty = csv_list[100:]

#First 100 calues
for row in firstHundred:
    if row != firstHundred[0]:
        x1 = np.append(x1,int(row[1]))
        x2 = np.append(x2,int(row[2]))
        x3 = np.append(x3,int(row[3]))
        x4 = np.append(x4,int(row[4]))
        x5 = np.append(x5,int(row[5]))
        x6 = np.append(x6,int(row[6]))
        y = np.append(y, int(row[7]))

#Last 20 x values
for row in lastTwenty:
    if row != lastTwenty[0]:
        x1a = np.append(x1a,int(row[1]))
        x2a = np.append(x2a,int(row[2]))
        x3a = np.append(x3a,int(row[3]))
        x4a = np.append(x4a,int(row[4]))
        x5a = np.append(x5a,int(row[5]))
        x6a = np.append(x6a,int(row[6]))

#Last 20 values
BaseTrainX = np.vstack((x1a,x2a,x3a,x4a,x5a,x6a)).T

#First 100 values
X = np.vstack((x1,x2,x3,x4,x5,x6)).T

#Cross validation for first 100
SplitX = np.array_split(X,[80,100])
trainX = SplitX[0]
testX = SplitX[1]

#Cross validation for first 100
SplitY = np.array_split(y,[80,100])
trainY = SplitY[0]
testY = SplitY[1]

#Multiple Linear Regression
MLR = LinearRegression()
MLR.fit(trainX,trainY)
MLRpred = MLR.predict(testX)

#Random Forest Regressor
RFR = RandomForestRegressor(n_estimators=10)
RFR.fit(trainX,trainY)
RFRpred = RFR.predict(testX)

#Decision Tree Regressor
DTR = DecisionTreeRegressor()
DTR.fit(trainX,trainY)
DTRpred = DTR.predict(testX)

#Calculating MSE for which is best
# Calculating r2 Score for which is best
print(f'Mean Squared error Multiple Linear Regression: {mean_squared_error(testY, MLRpred)}')
print(f'R Square for Multiple Linear : {r2_score(testY, MLRpred)}')
print(f'Mean Squared error Random Forest Regressor: {mean_squared_error(testY, RFRpred)}')
print(f'R Square Squared for Random Rorest: {r2_score(testY, RFRpred)}')
print(f'Mean Squared error Decision Tree Regressor: {mean_squared_error(testY, DTRpred)}')
print(f'R Square Squared for Decision Tree: {r2_score(testY, DTRpred)}')

#Random Forest Regressor is best for our data
# Because MSE is lower and r^2 is close to 1
RFR.fit(X,y)
RFRpred = RFR.predict(BaseTrainX)

#My 20 y values
print(RFRpred)
print(RFRpred[1])