# bike.py
# make a prediction for the Kaggle bike-sharing challenge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as systime
import datetime as dtime

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

#---define some useful functions
def splitDatetime(data): 
    sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
    date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
    time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
    data['year'] = date['year'].astype(int)
    data['month'] = date['month'].astype(int)
    data['day'] = date['day'].astype(int)
    data['hour'] = time['hour'].astype(int)
    data['weekday']=0 #weekday=0: monday, weekday=6: sunday.
    for i in range(0,len(train.year)):
        data.weekday[i]=dtime.date(train.year[i], train.month[i], train.day[i]).weekday()   
    return data

def normalize(data): #feature normalization
    data = (data - data.mean()) / (data.max() - data.min())
    return data

def predict(est, train, test, features, target):

    est.fit(train[features], train[target])

    with open("pred.csv", 'wb') as f:
        f.write("datetime,count\n")

        for index, value in enumerate(list(est.predict(test[features]))):
            f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))
            
def crossval(est, train, features, target): #cross-validation
    start = systime.time()    
    
    train_train, train_test, target_train, target_test = train_test_split(train[features], train[target], test_size=0.33, random_state=42)

    est.fit(train_train, target_train)

    end = systime.time()
    print 'rmsle:', rmsle(target_test, est.predict(train_test))
    print 'elapsed time:', end-start
    
def sle(actual, predicted): #squared log error
    return (np.power(np.log(np.array(actual)+1) - 
            np.log(np.array(predicted)+1), 2))
    
def rmsle(targets, predictions): #root mean squared log error
    return np.sqrt((sle(targets, predictions)**2).mean())


#---main analysis---
#read in the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = splitDatetime(train)
test = splitDatetime(test)

# determine the features to be used in the Machine Learning algorithm
target = 'count'
features = ['workingday','temp','atemp','humidity','windspeed','year','month','day','hour','weekday']

#define the ML algorithm
x=500
est = RandomForestRegressor(n_estimators=x) #one regressor for all

# crossvalidate the one-for-all regressor
crossval(est, train, features, 'casual')

# based on the estimator, evaluate the feature importances
importances = est.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking all:")
for f in range(len(features)):
    print("%d. feature: %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# make a prediction
predict(est, train, test, features, target)

# create output containing prediction for the test set
with open("pred.csv", 'wb') as f:
    f.write("datetime,count\n")
    for index, value in enumerate(list(test['count'])):
        f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))
