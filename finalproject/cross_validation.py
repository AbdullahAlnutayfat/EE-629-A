import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pdb
from pandas import *
from scipy import stats # for linear regression 
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
#********************************
data = read_csv('rpifinaldata.csv')
t = data['Date/Time']
x = data['CPU usage %']
y = data['Temperature C'] 
#********************************
f1 = plt.figure(1)

X = data.loc[:, 'CPU usage %':'Temperature C']
lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr,X,y,cv=10) 

plt.scatter(y,predicted,c='y',marker='o',edgecolors=(0,0,0))
plt.plot([y.min(),y.max()], [y.min(),y.max()], lw=3)

plt.xlabel('Temperature C')
plt.ylabel('predicted')
plt.title('Cross-Validation Prediction')
#***************Method two*********************
f2 = plt.figure(2)
# subtitute X with x.reshape(-1,1) CPU data only
lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr,x.reshape(-1,1),y,cv=10) 

plt.scatter(y,predicted,c='y',marker='o',edgecolors=(0,0,0))
plt.plot([y.min(),y.max()], [y.min(),y.max()], lw=3)

plt.xlabel('Temperature C')
plt.ylabel('predicted')
plt.title('Cross-Validation Prediction')
plt.show()
