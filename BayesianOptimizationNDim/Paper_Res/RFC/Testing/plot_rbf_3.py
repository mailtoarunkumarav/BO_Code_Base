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
from sklearn.ensemble import RandomForestClassifier

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
# iris = load_breast_cancer()
# X = iris.data
# y = iris.target

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
print("Working with wine Data")
winedata = pd.read_csv(url)
X = winedata.drop(winedata.columns[[0]], axis=1)
y = winedata.iloc[:, 0]

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# print("working with adult data")
# adult_data = pd.read_csv(url)
# X = adult_data.drop(adult_data.columns[[-1]], axis=1)
# y = adult_data.iloc[:, -1]

# import os
#
# files = os.listdir("../Dataset/Vehicle")
# total_data_frame = pd.DataFrame()
# for each_file in files:
#     each_data_frame= pd.read_csv("../Dataset/Vehicle/"+str(each_file),header=None, sep=r'[ ]', engine='python')
#     total_data_frame = pd.concat([total_data_frame,each_data_frame],ignore_index=True)
#
# X= total_data_frame.drop(total_data_frame.columns[[-1]], axis=1)
# y = total_data_frame.iloc[:, -1]
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# C_range = np.logspace(-3, 3, 7)
# gamma_range = np.logspace(-5, 0, 6)
# C_range = np.linspace(1e-15, 1e5, 50)
# gamma_range = np.linspace(1e-5, 1e0, 50)
# print(C_range, gamma_range)
# param_grid = dict(gamma=gamma_range, C=C_range)

esti_range = np.arange(1,50,1)
split_range = np.arange(2,200,1)
print(split_range)
# param_grid = dict(n_estimators=esti_range, min_samples_split=split_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# rfc = RandomForestClassifier()
# grid = GridSearchCV(rfc, param_grid=param_grid, cv=cv)
# grid.fit(X, y)
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
#
# scores = grid.cv_results_['mean_test_score'].reshape(len(split_range),
#                                                      len(esti_range))
#
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot
#            ,            norm=MidpointNormalize(vmin=0.2, midpoint=0.92)
#             )
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# plt.yticks(np.arange(len(split_range)), split_range)
# plt.xticks(np.arange(len(esti_range)), esti_range, rotation=45)
# plt.title('Validation accuracy')



e, s = np.meshgrid(esti_range, split_range)
print(e.shape, s.shape)
y = np.zeros((198,49))
print('Test')
max_accuracy = -1 * float('inf')
arr = []
for i in range(e.shape[0]):
    print(i)
    for j in range(e.shape[1]):

        rfc = RandomForestClassifier(n_estimators=e[i][j], min_samples_leaf=s[i][j], random_state=42)
        rfc.fit(X_train, y_train)
        accuracy = rfc.score(X_test, y_test)
        y[i][j]=accuracy
        if (accuracy > max_accuracy or accuracy == 1):

            if (accuracy > max_accuracy):
                print("\nMax: e", e[i][j], "\t s=", s[i][j], "\taccuracy=", accuracy)
            max_accuracy = accuracy
            arr.append(np.array([e[i][j], s[i][j], accuracy]))


# al = np.flip(-al, axis=1)
# y = np.flip(y, axis=1)
print(arr)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(e, s, y, rstride=1, cstride=1,cmap='viridis', linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)



plt.show()




