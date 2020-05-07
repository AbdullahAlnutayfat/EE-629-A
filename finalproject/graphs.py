import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pdb
from pandas import *
from scipy import stats # for linear regression 

data = read_csv('rpifinaldata.csv')
t = data['Date/Time']
# taking time only t[0][11:19]
x = data['CPU usage %']
y = data['Temperature C'] 
#l = [slop *i + intercept for i in x]
#************************************
f1 = plt.figure(1)
plt.plot(t, x,'b-', t, y,'r-')
plt.legend(('CPU','Temp'),loc='lower center', shadow=True)
plt.title('CPU usage % and Temperature C')
plt.xlabel('Time')
plt.ylabel('Time series for CPU and Temp')

ax = plt.gca()
ax.set_xticklabels(t, minor=False, rotation=50)
spacing = 50
ax.xaxis.set_major_locator(ticker.MultipleLocator(spacing))

#************************************
f2 = plt.figure(2)
cols = data.columns

n, bins, patches = plt.hist(x,20,normed=1, facecolor='b', alpha=0.5)
#n, bins, patches = plt.hist(x,facecolor='b', alpha=0.9)
plt.title('Histogram of CPU usage %')
plt.xlabel('CPU usage %')
plt.ylabel('probability')
plt.axis([0, 70, 0, 0.125])
#************************************

f3 = plt.figure(3)

n, bins, patches = plt.hist(y,20,normed=1, facecolor='r', alpha=0.5)
#n, bins, patches = plt.hist(x,facecolor='b', alpha=0.9)
plt.title('Histogram of Temperature C')
plt.xlabel('Temp C')
plt.ylabel('probability')
plt.axis([45, 70, 0, 0.3])

#************************************
# horizontal boxes
f4 = plt.figure(4)
plt.boxplot(x, 0, 'r*', 0,whis=.4)
plt.xlabel('CPU usage %')
plt.title('Horizontal Boxplot of CPU usage %')
# vertical boxes
f5 = plt.figure(5)
plt.boxplot(y,0,'r*',whis=.4)
plt.ylabel('Temperature C')
plt.title('Basic Boxplot of Temp C')
plt.axis([0, 2, 45, 70])

#************************************
f6 = plt.figure(6)

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
l = [slope *i + intercept for i in x]
#l = x*slope+intercept
plt.plot(x,y,'go',x,l,'r')
#plt.plot([intercept, intercept+slope])
#plt.plot(l)
plt.xlabel('CPU usage %')
plt.ylabel('Temperature C')
plt.title('Linear Regression')
#************************************
'''
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

f7 = plt.figure(7)
y1 = y
#X = np.random.rand(345,2)
#X = x*y
#X = X.reshape(345,2)
#X = np.random.randint(70, size=(345, 2))
X = data.loc[:, 'CPU usage %':'Temperature C']
lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr,X,y=y1,cv=10) #cv=10
#plt.plot(y,x,'yo')
plt.scatter(y,predicted,c='y',marker='o',edgecolors=(0,0,0))
plt.plot([y1.min(),y1.max()], [y1.min(),y1.max()], lw=3)

plt.xlabel('Temperature C')
plt.ylabel('predicted')
plt.title('Cross-Validation Prediction')
'''
#************************************
'''
f8 = figure(8)
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
#y = boston.target we will use above y
#print('Number of instances: %d' % (boston.data.shape[0]))

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston.data, y, cv=10)


scatter(y, predicted)
plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=2)
set_xlabel('Measured')
set_ylabel('Predicted')
'''

#************************************

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

f8 = plt.figure(8)
y1 = y
#X = np.random.rand(345,2)
#X = x*y
#X = X.reshape(345,2)
#X = np.random.randint(70, size=(345, 2))
X = data.loc[:, 'CPU usage %':'Temperature C']
lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr,x.reshape(-1,1),y=y1,cv=10) #cv=10
#plt.plot(y,x,'yo')
plt.scatter(y,predicted,c='y',marker='o',edgecolors=(0,0,0))
plt.plot([y1.min(),y1.max()], [y1.min(),y1.max()], lw=3)

plt.xlabel('Temperature C')
plt.ylabel('predicted')
plt.title('Cross-Validation Prediction')
#************************************
plt.show()

