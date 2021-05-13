import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC




class PreProcessor:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_dataset(self):
        dataset_input_file = "Dataset/wdbc.csv"
        print("working with Breast-Cancer data")

        # Wine data
        # winedata = pd.read_csv(url)
        # D = winedata.drop(winedata.columns[[0]], axis=1)
        # f = winedata.iloc[:, 0]

        total_data = pd.read_csv(dataset_input_file)

        D = total_data.drop(total_data.columns[0], axis=1)
        f = total_data.iloc[:, 0]

        #Method1
        min_max_scaler = MinMaxScaler()
        D_std = min_max_scaler.fit_transform(D)

        # #Method2
        # cols = np.arange(0, len(D.iloc[0]))
        # D_std = self.normalise_numerical_columns(D, cols)

        #Categorical Toy Data
        # D = [['num1', 'cat1', 'cat2', 'num2', 'num3'],
        #     [1.2, 'Arun', 'Anjana', 92.96, 83.5],
        #      [8, 'Arun', 'Anjana', 56, 24.5],
        #      [0.1, 'Arun', 'Anjana', 2, 2],
        #      [2, 'Arun', 'Anjana', 100, 100]
        #      ]
        # f = [[1],
        #     [-1],
        #      [1],
        #      [-1],
        #      [1]]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(D_std, f, test_size=0.20)

    def start_pp(self):
        self.read_dataset()
        # self.scaler()
        # self.column_transformers()
        self.min_max_scaler()

    def normalise_numerical_columns(self, D, cols):

        for each_col in cols:
            max_val = max(D.iloc[:, each_col])
            min_val = min(D.iloc[:, each_col])
            D.iloc[:, each_col] = (D.iloc[:, each_col] - min_val)/(max_val - min_val)

        return D


    def min_max_scaler(self):
        min_max_scaler = MinMaxScaler()
        whole_dataset = pd.concat([self.X_train, self.X_test])
        print(len(whole_dataset))
        min_max_scaler.fit_transform(whole_dataset)
        X_train_minmax = min_max_scaler.fit_transform(self.X_train)
        # X_test_minmax = min_max_scaler.fit_transform(self.X_test)
        print(self.X_train)
        print(max(whole_dataset.iloc[:,0]), min(whole_dataset.iloc[:,0]), X_train_minmax,"\n\n")

        # print(self.X_test,X_test_minmax)

        # new_xtrain = self.X_train.iloc[0:5]
        # X_train_minmax = min_max_scaler.fit_transform(new_xtrain)
        # X_test_minmax = min_max_scaler.fit_transform(self.X_test)
        #
        # for i in range(len(new_xtrain)):
        #     print(new_xtrain.iloc[i, 0:2])
        #
        # print(self.X_train.iloc, X_train_minmax[:, 0:2])


    def column_transformers(self):
        numeric_features = ['num1', 'num2', 'num3']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['cat1', 'cat2']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', SVC())])


        clf.fit(self.X_train, self.y_train)
        print("model score: %.3f" % clf.score(self.X_test, self.y_test))

    def scaler(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        X_train_std = sc.transform(self.X_train)
        X_test_std = sc.transform(self.X_test)

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((self.y_train, self.y_test))
        print(self.X_train.iloc[0])
        # total_data = pd.read_csv(file)
        # data = total_data.drop(total_data.columns[0], axis=1)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(data)
        # data_new = enc.transform(data)
        # f = open("out", "w")
        # f.write(str(data_new))
        # f.flush()

if __name__ == '__main__':
    data_pp = PreProcessor()
    data_pp.start_pp()

