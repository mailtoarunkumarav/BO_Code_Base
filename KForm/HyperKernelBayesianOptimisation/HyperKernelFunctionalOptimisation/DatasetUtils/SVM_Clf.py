from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


# def my_kernel(data1, data2):
#     kernel_mat = np.dot(data1, data2.T)
#     return kernel_mat
np.random.seed(300)

# file = "Dataset/wdbc.csv"
# total_data = pd.read_csv(file)
# D = total_data.drop(total_data.columns[0], axis=1)
# f = total_data.iloc[:, 0]
#
# D_train, D_test, y_train, y_test = train_test_split(D, f, test_size=0.20)
#
# svc = SVC(kernel=my_kernel)
# svc.fit(D_train, y_train)
# #kernel_test = np.dot(X_test, X_train[svc.support_, :].T)
# y_pred = svc.predict(y_test)
# print ('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))

def mykernel(data1, data2):
    kernel_mat = np.dot(data1, data2.T)
    return kernel_mat

# digits = load_digits()
# X, y = shuffle(digits.data, digits.target)
# X_train, X_test = X[:1000, :], X[100:, :]
# y_train, y_test = y[:1000], y[100:]

# file = "Dataset/wdbc.csv"
# total_data = pd.read_csv(file)
# D = total_data.drop(total_data.columns[0], axis=1)
# f = total_data.iloc[:, 0]
# X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)

# file = "Dataset/iris.csv"
# total_data = pd.read_csv(file)
# f = total_data.iloc[:, -1]
# D = total_data.drop(total_data.columns[-1], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)

# file = "Dataset/glass.csv"
# total_data = pd.read_csv(file)
# f = total_data.iloc[:, -1]
# D = total_data.drop(total_data.columns[-1], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)


file = "Dataset/credit.csv"
total_dataframe = pd.read_csv(file, na_values='?')
dataframe = total_dataframe.fillna(method='bfill')

f = dataframe.iloc[:, -1]
dataframe = dataframe.drop(dataframe.columns[-1], axis=1)

dataframe_num = dataframe.select_dtypes(include=[np.number])
print(dataframe_num)
min_max_scaler = MinMaxScaler()
dataframe_num_std = min_max_scaler.fit_transform(dataframe_num)
print(dataframe_num_std)

dataframe_cat = dataframe.select_dtypes(include=[object])
label_encoder = preprocessing.LabelEncoder()
labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
#
onehot_encoder = preprocessing.OneHotEncoder()
onehot_encoder.fit(labeled_dataframe_cat)
onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()

D = np.concatenate((dataframe_num_std, onehotlabel_cols), axis=1)


X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))
exit(0)

# file = "Dataset/credit2.csv"
# dataframe = pd.read_csv(file, na_values='?')
# Dataframe = dataframe.fillna(method='ffill')
# print(dataframe)
# f = dataframe.iloc[:, -1]
# D = dataframe.drop(dataframe.columns[-1], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)
#

# svc = SVC(kernel='linear')
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
# print('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))



# # svc = SVC(kernel=mykernel, random_state=42)
# svc = SVC(random_state=42)
#
# # kernel_train = np.dot(X_train, X_train.T)  # linear kernel
# # svc.fit(kernel_train, y_train)
# svc.fit(X_train, y_train)
#
# #kernel_test = np.dot(X_test, X_train[svc.support_, :].T)
# # kernel_test = np.dot(X_test, X_train.T)
# # y_pred = svc.predict(kernel_test)
# y_pred = svc.predict(X_test)
# print how our model looks after hyper-parameter tuning

svc = SVC(random_state=42, kernel='linear')
# # grid_values = {'C': [10, 100, 1000, 10000, 100000, 1000000, 10000000]}
grid_values = {'C': [0.001, 0.01, 0.1, 1, 10,100,1000,10000]
               , 'gamma': [0.001, 0.01, 0.1, 1, 10, 1000, 10000]
               }

grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=3)
grid_svm_acc.fit(X_train, y_train)
print("Best Params: ", grid_svm_acc.best_params_)
print("Best estimator: ", grid_svm_acc.best_estimator_)
print("Best estimator: ", grid_svm_acc.best_score_)
y_pred = grid_svm_acc.predict(X_test)
print('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))
