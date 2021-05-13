import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.optimize as opt
from scipy.stats import norm

import scipy.optimize as opt
import scipy as sp
import matplotlib.pyplot as plt
# from numpy import *
import numpy as np
import datetime
from scipy.spatial import distance
import collections
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import plotly.plotly as py
import plotly.graph_objs as go
# from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import SGDClassifier
import pandas as pd


np.random.seed(300)

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search

iris = load_iris()
# iris = load_breast_cancer()
X = iris.data
y = iris.target

# scaler = StandardScaler()
# scaler.fit(X, y)
# X = scaler.transform(X)


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# # bcdata = pd.read_csv("Dataset/wdbc.data")
# print("Working with WDBC Data")
# bcdata = pd.read_csv(url)
# X = bcdata.drop(bcdata.columns[[0, 1]], axis=1)
# y = bcdata.iloc[:, 1]

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# print("Working with wine Data")
# winedata = pd.read_csv(url)
# X = winedata.drop(winedata.columns[[0]], axis=1)
# y = winedata.iloc[:, 0]


import os
#
# files = os.listdir("../Dataset/Vehicle")
# total_data_frame = pd.DataFrame()
# for each_file in files:
#     each_data_frame= pd.read_csv("../Dataset/Vehicle/"+str(each_file),header=None, sep=r'[ ]', engine='python')
#     total_data_frame = pd.concat([total_data_frame,each_data_frame],ignore_index=True)
# X= total_data_frame.drop(total_data_frame.columns[[-1]], axis=1)
# y = total_data_frame.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

l1_ratio_range = np.linspace(0.01,1,20)
# alpha_range = np.logspace(-10, -1, 10)
# l1_ratio_range = np.logspace(-5,0,6)
alpha_range = np.logspace(-7, -1, 20)

print( "l1ratio = ", l1_ratio_range,"alp = ", alpha_range)

param_grid = dict(alpha = alpha_range , l1_ratio = l1_ratio_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state= 42)
sgd = SGDClassifier(verbose=0,penalty='elasticnet', random_state= 42)
grid = GridSearchCV(sgd, param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# scores = grid.cv_results_['mean_test_score'].reshape(len(alpha_range),
#                                                      len(l1_ratio_range))
#
# plt.figure(figsize=(8, 6))
# # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot
#            # ,norm=MidpointNormalize(vmin=0.2, midpoint=0.92)
#            )
# plt.xlabel('l1ratio')
# plt.ylabel('alpha')
# plt.colorbar()
# plt.xticks(np.arange(len(l1_ratio_range)),l1_ratio_range , rotation=45)
# plt.yticks(np.arange(len(alpha_range)),alpha_range )
# plt.title('Validation accuracy')

# alpha_range = np.linspace(1, 7, 20)
alpha_range = np.linspace(-7, -1, 20)
al, l1 = np.meshgrid(alpha_range, l1_ratio_range)
print(al.shape, l1.shape)
y = np.zeros((20,20))
print('Test')
for i in range(al.shape[0]):
    for j in range(l1.shape[1]):
        # alp = 1.0 / np.power(10, al[i][j])
        alp = (10**(al[i][j]))
        sgclassifier = SGDClassifier(alpha=alp, l1_ratio=l1[i][j], penalty='elasticnet', random_state=42)
        sgclassifier.fit(X_train, y_train)
        accuracy = sgclassifier.score(X_test, y_test)
        y[i][j]=accuracy
        print(l1[i][j], alp, accuracy)

# al = np.flip(-al, axis=1)
# y = np.flip(y, axis=1)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(al, l1, y, rstride=1, cstride=1,cmap='viridis', linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)

# plt.contourf(al,l1,y, cmap='viridis')
# plt.colorbar()

# exit(0)

# al = 10**-4.24073551
# sgclassifier = SGDClassifier(alpha= al , l1_ratio=  0.31190913 , verbose=0, penalty='elasticnet', random_state= 42)
# sgclassifier.fit(X_train, y_train)
# y_pred = sgclassifier.predict(X_test)
# accuracy = sgclassifier.score(X_test, y_test)
# print(accuracy)
# exit(0)
alpha_range = np.logspace(-7, -1, 20)
max_accuracy = -1 * float('inf')
arr = []
for l1 in l1_ratio_range:
    for alp in alpha_range:
        sgclassifier = SGDClassifier(alpha=alp, l1_ratio=l1,penalty='elasticnet', random_state= 42)
        sgclassifier.fit(X_train, y_train)
        accuracy = sgclassifier.score(X_test, y_test)
        # print(l1,alp, accuracy)
        if (accuracy > max_accuracy or accuracy == 1):

            if (accuracy > max_accuracy):
                print("\nMax: l1", l1, "\t alp=", alp, "\taccuracy=", accuracy)
            max_accuracy = accuracy
            arr.append(np.array([l1, alp, accuracy]))
            # print(confusion_matrix(y_test,y_pred))
            # print("\n\nclassification report ", classification_report(y_test, y_pred))
print("\n\n\n\n\n\n",arr)
timenow = datetime.datetime.now()
print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

plt.show()




'''
# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search

# iris = load_iris()
iris = load_breast_cancer()
X = iris.data
y = iris.target
#
# scaler = StandardScaler()
# scaler.fit(X, y)
# X = scaler.transform(X)


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# # bcdata = pd.read_csv("Dataset/wdbc.data")
# print("Working with WDBC Data")
# bcdata = pd.read_csv(url)
# X = bcdata.drop(bcdata.columns[[0, 1]], axis=1)
# y = bcdata.iloc[:, 1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

l1_ratio_range = np.linspace(0.01,1,20)
# alpha_range = np.logspace(-10, -1, 10)
# l1_ratio_range = np.logspace(-5,0,6)
alpha_range = np.logspace(-7, -1, 20)

print( "l1ratio = ", l1_ratio_range,"alp = ", alpha_range)

param_grid = dict(alpha = alpha_range , l1_ratio = l1_ratio_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state= 42)
sgd = SGDClassifier(verbose=0,penalty='elasticnet', random_state= 42)
grid = GridSearchCV(sgd, param_grid=param_grid, cv=cv)
grid.fit(X, y)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
#
# scores = grid.cv_results_['mean_test_score'].reshape(len(alpha_range),
#                                                      len(l1_ratio_range))
#
# plt.figure(figsize=(8, 6))
# # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot
#            # ,norm=MidpointNormalize(vmin=0.2, midpoint=0.92)
#            )
# plt.xlabel('l1ratio')
# plt.ylabel('alpha')
# plt.colorbar()
# plt.xticks(np.arange(len(l1_ratio_range)),l1_ratio_range , rotation=45)
# plt.yticks(np.arange(len(alpha_range)),alpha_range )
# plt.title('Validation accuracy')

alpha_range = np.linspace(1, 7, 20)
al, l1 = np.meshgrid(alpha_range, l1_ratio_range)
print(al.shape, l1.shape)
y = np.zeros((20,20))
print('Test')
for i in range(al.shape[0]):
    for j in range(l1.shape[1]):
        alp = 1.0 / np.power(10, al[i][j])
        sgclassifier = SGDClassifier(alpha=alp, l1_ratio=l1[i][j], penalty='elasticnet', random_state=42)
        sgclassifier.fit(X_train, y_train)
        accuracy = sgclassifier.score(X_test, y_test)
        y[i][j]=accuracy
        print(l1[i][j], alp, accuracy)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(al, l1, y, rstride=1, cstride=1,cmap='viridis', linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)

# plt.contourf(al,l1,y, cmap='viridis')
# plt.colorbar()

# exit(0)

# al = 10**-4.24073551
# sgclassifier = SGDClassifier(alpha= al , l1_ratio=  0.31190913 , verbose=0, penalty='elasticnet', random_state= 42)
# sgclassifier.fit(X_train, y_train)
# y_pred = sgclassifier.predict(X_test)
# accuracy = sgclassifier.score(X_test, y_test)
# print(accuracy)
# exit(0)
alpha_range = np.logspace(-7, -1, 20)
max_accuracy = -1 * float('inf')
arr = []
for l1 in l1_ratio_range:
    for alp in alpha_range:
        sgclassifier = SGDClassifier(alpha=alp, l1_ratio=l1,penalty='elasticnet', random_state= 42)
        sgclassifier.fit(X_train, y_train)
        accuracy = sgclassifier.score(X_test, y_test)
        print(l1,alp, accuracy)
        if (accuracy > max_accuracy or accuracy == 1):

            if (accuracy > max_accuracy):
                print("\nMax: l1", l1, "\t alp=", alp, "\taccuracy=", accuracy)
            max_accuracy = accuracy
            arr.append(np.array([l1, alp, accuracy]))
            # print(confusion_matrix(y_test,y_pred))
            # print("\n\nclassification report ", classification_report(y_test, y_pred))
print("\n\n\n\n\n\n",arr)
timenow = datetime.datetime.now()
print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

plt.show()





'''

