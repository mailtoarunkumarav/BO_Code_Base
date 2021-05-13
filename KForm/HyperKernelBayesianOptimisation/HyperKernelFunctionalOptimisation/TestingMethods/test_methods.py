import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

strg = "a\c"
new_strng = strg.replace("\\","")
print(new_strng)

print(np.random.normal(0,0.001,5))
exit(0)

# Commenting for regression data - Forestfire

number_of_dimensions = 2
bounds = [[0,1], [0,1]]
number_of_observed_samples = 20
random_points = []
X = []

# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                   number_of_observed_samples).reshape(1, number_of_observed_samples)
    random_points.append(random_data_point_each_dim)

# Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
random_points = np.vstack(random_points)

# Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
for sample_num in np.arange(number_of_observed_samples):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)

print(X)

X = np.array([
    [66,    50],
    [70,    50],
    [69,    50],
    [68,    50],
    [67,    50],
    [72,     50],
    [73,   100],
    [70,    100],
    [57,    200],
    [63,    200],
    [70,    200],
    [78,    200],
    [67,    200],
    [53,    200],
    [67,    200],
    [75,    200],
    [70,    200],
    [81,    200],
    [76,    200],
    [79,    200],
    [75,    200],
    [76,    200],
   [58,    200]])

print(X)
exit(0)
import test2

test2.start_testing()
print("hello")
exit(0)

# bcdata = pandas.read_csv('wdbc.data')
# X = bcdata.drop(bcdata.columns[[0, 1]], axis=1)
# y = bcdata.iloc[:, 1]

colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pandas.read_csv("iris.data", names=colnames)
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

# X = dataset.drop(dataset.columns[2])
# y = dataset.iloc[:, 13]

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=50, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))

    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))

print(np.mean(scores))
