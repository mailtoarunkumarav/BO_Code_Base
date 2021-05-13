import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV


import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH


class SVM_Wrapper:

    def __init__(self):
        self.dataset_input_file = None
        self.X_train = None
        self.Xtest = None
        self.y_train = None
        self.y_test = None
        self.D = None
        self.f = None

    def construct_svm_classifier(self, dataset):

        if dataset == "WDBC":
            self.dataset_input_file = "..\DatasetUtils\Dataset\wdbc.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            D = total_data.drop(total_data.columns[0], axis=1)

            #Method 1
            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

            # Method2
            # cols = np.arange(0, len(D.iloc[0]))
            # D_std = self.normalise_numerical_columns(D, cols)

            f = total_data.iloc[:, 0]

        elif dataset == "IRIS":
            self.dataset_input_file = "..\DatasetUtils\Dataset\iris.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "GLASS":
            self.dataset_input_file = "..\glass.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "IONOS":
            self.dataset_input_file = "..\DatasetUtils\Dataset\ionosphere.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)
            # print(D, "\n\n", D_std)
            # exit(0)

        elif dataset == "SONAR":
            self.dataset_input_file = "..\DatasetUtils\Dataset\sonar.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "HEART":
            self.dataset_input_file = "..\DatasetUtils\Dataset\heart.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "CREDIT":
            self.dataset_input_file = "..\DatasetUtils\Dataset\credit.csv"
            total_dataframe = pd.read_csv(self.dataset_input_file, na_values='?')
            dataframe = total_dataframe.fillna(method='bfill')

            f = dataframe.iloc[:, -1]
            dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
            D = dataframe

            dataframe_num = dataframe.select_dtypes(include=[np.number])
            min_max_scaler = MinMaxScaler()
            dataframe_num_std = min_max_scaler.fit_transform(dataframe_num)

            dataframe_cat = dataframe.select_dtypes(include=[object])
            label_encoder = preprocessing.LabelEncoder()
            labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
            #
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(labeled_dataframe_cat)
            onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()

            D_std = np.concatenate((dataframe_num_std, onehotlabel_cols), axis=1)

        elif dataset == "GERMAN_CREDIT":
            self.dataset_input_file = "..\DatasetUtils\Dataset\German_Credit.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "SEED":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\Seed_Data.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "PIMA":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\diabetes.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "DERMATOLOGY":
            self.dataset_input_file = "..\DatasetUtils\Dataset\dermatology.csv"
            total_dataframe = pd.read_csv(self.dataset_input_file, na_values='?')
            dataframe = total_dataframe.fillna(method='bfill')

            f = dataframe.iloc[:, -1]
            dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
            D = dataframe

            dataframe_cat = dataframe.select_dtypes(include=[np.number])
            label_encoder = preprocessing.LabelEncoder()
            labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
            #
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(labeled_dataframe_cat)
            onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()
            D_std = onehotlabel_cols

        elif dataset == "WINE":
            self.dataset_input_file = "..\DatasetUtils\Dataset\Wine.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "BIO":
            self.dataset_input_file = "..\DatasetUtils\Dataset\data_biodeg.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "CONTRA":
            self.dataset_input_file = "..\DatasetUtils\Dataset\dataset_contra.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "PHO":
            self.dataset_input_file = "..\DatasetUtils\Dataset\dataset_phoneme.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "HAY":
            self.dataset_input_file = "..\DatasetUtils\Dataset\dataset_hayes.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "ECO":
            self.dataset_input_file = "..\DatasetUtils\Dataset\dataset_ecoli.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "CAR":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_car.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        # Store and Split the dataset
        self.D = D
        self.f = f
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(D_std, f, test_size=0.20)

    def compute_kerenel_mat_hyperk(self, data_point1, data_point2, kernel_bias, basis_weights, kernel_samples, hypergp_obj):
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                # dp1 = data_point1.iloc[i].to_numpy().reshape(1, -1)
                dp1 = data_point1[i]
                # dp2 = data_point2.iloc[j].to_numpy().reshape(1, -1)
                dp2 = data_point2[j]
                each_kernel_val_KFO = hypergp_obj.estimate_kernel_for_Xtil(dp1, dp2, kernel_bias, basis_weights, kernel_samples)
                # each_kernel_val = hyperGP_obj.estimate_kernel_for_Xtil(data_point1[i, :], data_point2[j, :], observations_kernel)
                # num = self.hyper_gp_obj.estimate_kernel_for_Xtil(data_point1[i, :], data_point2[j, :], observations_kernel)
                # den1 = self.hyper_gp_obj.estimate_kernel_for_Xtil(data_point1[i, :], data_point1[i, :], observations_kernel)
                # den2 = self.hyper_gp_obj.estimate_kernel_for_Xtil(data_point2[j, :], data_point2[j, :], observations_kernel)
                # each_kernel_val = num / (np.sqrt(den1*den2))

                # new MKL implementation
                l = 0.2
                sig = 1
                difference = (dp1 - dp2)
                l2_difference_sq = np.dot(difference, difference.T)
                each_kernel_val_SE = (sig ** 2) * (np.exp((-1 / (2 * (l**2))) * l2_difference_sq))
                each_kernel_val_lin = np.dot(dp1, dp2)

                kernel_mat[i, j] = each_kernel_val_KFO + basis_weights[3] * each_kernel_val_SE + basis_weights[4] * each_kernel_val_lin
        # self.hyper_gp_obj.pre_calculated_L_Kappa_Kobs = None # Not required with EVD method of computation
        return kernel_mat

    def normalise_numerical_columns(self, D, cols):

        for each_col in cols:
            max_val = max(D.iloc[:, each_col])
            min_val = min(D.iloc[:, each_col])
            D.iloc[:, each_col] = (D.iloc[:, each_col] - min_val)/(max_val - min_val)
        return D

    def compute_accuracy(self, kernel_type, kernel_bias, basis_weights, kernel_samples, hyperGP_obj):

        print("\nComputing accuracy for the kernel... ", "\nConstructing SVM Classifier")
        # return np.array([[np.random.uniform(0,1)]])

        current_observations_kernel = basis_weights[0] * kernel_bias + basis_weights[1] * kernel_samples[0] + basis_weights[2] * \
                                 kernel_samples[1]

        # Grid search
        grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                       # ,'gamma': [0.001, 0.01, 0.1, 1, 10, 1000, 10000]
                       }
        svc = SVC(kernel='precomputed', random_state=42)
        grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=1)
        PH.printme(PH.p1, "Computing Xtr_Xtr .... ")
        kernel_mat_Xtr_Xtr = self.compute_kerenel_mat_hyperk(self.X_train, self.X_train, kernel_bias, basis_weights, kernel_samples,
                                                             hyperGP_obj)
        PH.printme(PH.p1, "Fitting SVM for the Data .... ")
        grid_svm_acc.fit(kernel_mat_Xtr_Xtr, self.y_train)
        PH.printme(PH.p1, "Computing Xte_Xtr .... ")
        kernel_mat_Xte_Xtr = self.compute_kerenel_mat_hyperk(self.X_test, self.X_train, kernel_bias, basis_weights, kernel_samples,
                                                             hyperGP_obj)
        PH.printme(PH.p1, "Predicting Values...")
        y_pred = grid_svm_acc.predict(kernel_mat_Xte_Xtr)
        accuracy = accuracy_score(self.y_test, y_pred)
        PH.printme(PH.p1, "Accuracy: ", accuracy)
        return current_observations_kernel, np.array([[accuracy]])

        # svc = SVC(kernel='precomputed', random_state=42)
        # PH.printme(PH.p1, "Computing Xtr_Xtr .... ")
        # kernel_mat_Xtr_Xtr = self.compute_kerenel_mat_hyperk(self.X_train, self.X_train, observations_kernel, hyperGP_obj)
        # PH.printme(PH.p1, "Fitting SVM for the Data .... ")
        # svc.fit(kernel_mat_Xtr_Xtr, self.y_train)
        # PH.printme(PH.p1, "Computing Xte_Xtr .... ")
        # kernel_mat_Xte_Xtr = self.compute_kerenel_mat_hyperk(self.X_test, self.X_train, observations_kernel, hyperGP_obj)
        # PH.printme(PH.p1, "Predicting Values...")
        # y_pred = svc.predict(kernel_mat_Xte_Xtr)
        # accuracy = accuracy_score(self.y_test, y_pred)
        # PH.printme(PH.p1, "Accuracy: ", accuracy)
        # return np.array([[accuracy]])

    # # Method2 - Not required
    # def compute_kerenel_mat_hyperk_arx(self, data1, data2):
    #     kernel_mat = np.dot(data1, data2.T)
    #     return kernel_mat
    # def compute_accuracy_arx(self, kernel_type, observations_kernel, optional_hypergp_obj):
    #     svc = SVC(kernel=self.compute_kerenel_mat_hyperk)
    #     svc.fit(self.X_train, self.y_train)
    #     y_pred = self.svc.predict(self.X_test)
    #     accuracy = accuracy_score(self.y_test, y_pred)
    #     PH.printme(PH.p1, 'accuracy score: %0.3f' % accuracy)
    #     return np.array([[accuracy]])

