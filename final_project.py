import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import mean_absolute_error
from sklearn import tree

#load dataset
df = pd.read_csv("cleaned_vehicles.csv")
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
training = df
# it looks like ID is not useful
training = training.drop("id", axis = 1)

#training = training.drop("region", axis = 1)
#training = training.drop("manufacturer", axis = 1)
training = training.drop("model", axis = 1)
#training = training.drop("condition", axis = 1)
#training = training.drop("cylinders", axis = 1)
training = training.drop("fuel", axis = 1)
#training = training.drop("odometer", axis = 1)
#training = training.drop("title_status", axis = 1)
training = training.drop("transmission", axis = 1)
#training = training.drop("drive", axis = 1)
#training = training.drop("size", axis = 1)
#training = training.drop("type", axis = 1)
training = training.drop("paint_color", axis = 1)
training = training.drop("lat", axis = 1)
training = training.drop("long", axis = 1)

#training = training.sample(frac = 0.1)

#print(Y)

#print(training)

region = pd.get_dummies(training['region'])
#print(uc)
training = training.join(region, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("region", axis = 1)

manufacturer = pd.get_dummies(training['manufacturer'])
#print(uc)
training = training.join(manufacturer, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("manufacturer", axis = 1)
'''
model = pd.get_dummies(training['model'])
#print(uc)
training = training.join(model, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("model", axis = 1)
'''

condition = pd.get_dummies(training['condition'])
#print(uc)
training = training.join(condition, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("condition", axis = 1)

cylinders = pd.get_dummies(training['cylinders'])
#print(uc)
training = training.join(cylinders, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("cylinders", axis = 1)

'''
fuel = pd.get_dummies(training['fuel'])
#print(uc)
training = training.join(fuel, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("fuel", axis = 1)
'''

title_status = pd.get_dummies(training['title_status'])
#print(uc)
training = training.join(title_status, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("title_status", axis = 1)
'''
transmission = pd.get_dummies(training['transmission'])
#print(uc)
training = training.join(transmission, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("transmission", axis = 1)
'''
drive = pd.get_dummies(training['drive'])
#print(uc)
training = training.join(drive, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("drive", axis = 1)

size = pd.get_dummies(training['size'])
#print(uc)
training = training.join(size, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("size", axis = 1)

type = pd.get_dummies(training['type'])
#print(uc)
training = training.join(type, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("type", axis = 1)
'''
paint_color = pd.get_dummies(training['paint_color'])
#print(uc)
training = training.join(paint_color, lsuffix='_o', rsuffix='_d')
#print(training)
training = training.drop("paint_color", axis = 1)
'''
X = training
ct = datetime.datetime.now()
print("Right before get_dummies - current time:-", ct)

ct = datetime.datetime.now()
print("After dummies and before splitting - current time:-", ct)

#X.apply(pd.to_numeric)
#Y.apply(pd.to_numeric)

Y = training["price"]
training = training.drop("price", axis = 1)
#print(Y)
X = X.reset_index()
Y = Y.reset_index()
X = X.drop("index", axis = 1)
Y = Y.drop("index", axis = 1)
#print(Y)
for i in range(len(Y)):
    val = Y.at[i, "price"]
    # 70%
    if val <= 10000:
        Y.at[i, "price"] = 5000
    if 10000 < val < 20000:
        Y.at[i, "price"] = 15000
    if 20000 < val < 30000:
        Y.at[i, "price"] = 25000
    if 30000 < val < 40000:
        Y.at[i, "price"] = 35000
    if 40000 < val < 50000:
        Y.at[i, "price"] = 45000
    if 50000 < val < 60000:
        Y.at[i, "price"] = 55000
    if 60000 < val < 70000:
        Y.at[i, "price"] = 65000
    if 70000 < val < 80000:
        Y.at[i, "price"] = 75000
    if 80000 < val < 90000:
        Y.at[i, "price"] = 85000
    if val > 100000:
        Y.at[i, "price"] = 105000

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

ct = datetime.datetime.now()
print("After splitting - current time:-", ct)

classifier = DecisionTreeClassifier()

#X_train.apply(pd.to_numeric)
#X_test.apply(pd.to_numeric)
#y_train.apply(pd.to_numeric)
#y_test.apply(pd.to_numeric)

#X_train = X_train.reset_index()
#X_test = X_test.reset_index()
#y_train = y_train.reset_index()
#y_test = y_test.reset_index()

classifier.fit(X_train, y_train)

ct = datetime.datetime.now()
print("After fitting - current time:-", ct)

y_pred = classifier.predict(X_test)

ct = datetime.datetime.now()
print("After pred - current time:-", ct)

print("Y TEST:")
print(y_test)
print("Y PRED")
print(y_pred)

# Metrics
'''
acc = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
'''
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

score=r2_score(y_test,y_pred)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_pred))
print("root_mean_squared error of is=",np.sqrt(mean_squared_error(y_test,y_pred)))
