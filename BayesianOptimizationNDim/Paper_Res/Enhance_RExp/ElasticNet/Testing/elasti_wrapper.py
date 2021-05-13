from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,scorer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics

import time
from sklearn.metrics import mean_squared_error,accuracy_score

class Custom_Print(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

# To fix the random number genration, currently not able, so as to retain the random selection of points
random_seed = 300
np.random.seed(random_seed)

# Class for starting Bayesian Optimization with the specified parameters
class BayesianOptimizationWrapper:

    def opt_wrapper(start_time):


        print("\n###################################################################\n")
        timenow = datetime.datetime.now()
        print("Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # # Bank authentication
        # bankdata = pd.read_csv("Dataset/bill_authentication.csv")
        # X = bankdata.drop('Class', axis=1)
        # y = bankdata['Class']

        # #Iris data
        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        # # Assign colum names to the dataset
        # colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        # # Read dataset to pandas dataframe
        # #irisdata = pd.read_csv(url, names=colnames)
        # irisdata = pd.read_csv("../Dataset/iris.data", names=colnames)
        # X = irisdata.drop('Class', axis=1)

        # y = irisdata['Class']
        # for ind in np.arange(len(y)):
        #     if y.iloc[ind] =='Iris-virginica':
        #         y[ind]=0
        #     if y.iloc[ind] =='Iris-setosa':
        #         y[ind]=1
        #     if y.iloc[ind] == 'Iris-versicolor':
        #         y[ind] = 2

        # # # # Breast cancer data
        # colnames = [ 'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes','node-caps', 'deg-malig', 'breast','breast-quad','irradiat' ]
        # bcdata = pd.read_csv("Dataset/breast-cancer.data", names=colnames)
        # X = bcdata.drop('Class', axis=1)
        # y = bcdata['Class']


        # # # # Breast cancer wisconsin data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        # bcdata = pd.read_csv("Dataset/wdbc.data")
        bcdata = pd.read_csv(url)
        X = bcdata.drop(bcdata.columns[[0,1]], axis=1)
        y = bcdata.iloc[:,1]

        for ind in np.arange(len(y)):
            if y.iloc[ind] =='M':
                y[ind]=0
            if y.iloc[ind] =='B':
                y[ind]=1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


        elastic = linear_model.ElasticNet(normalize=True,max_iter=5000)
        search = GridSearchCV(estimator=elastic, param_grid={'alpha': np.logspace(-5, -2, 100), 'l1_ratio': np.logspace(-5, -2, 100)},
                              scoring='auc',n_jobs=1, refit=True, cv=10)
        search.fit(X, y)
        print(search.best_params_)


        alpha_values = np.logspace(-5, -2, 100)
        l1_penalty = np.logspace(-5, -2, 100)
        # scaler = preprocessing.StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        max_score = -1 * float('inf')
        arr=[]
        for alpha in alpha_values:
            for l1_p in l1_penalty:

                elastic = linear_model.ElasticNet(normalize=True, alpha=alpha, l1_ratio=l1_p)
                elastic.fit(X_train, y_train)
                # error = (mean_squared_error(y_true=y_test, y_pred=elastic.predict(X_test)))
                # score = elastic.score(elastic.predict(X_test), y_test)
                score = elastic.score(X_test, y_test)

                if(score > max_score):
                        print("\nMax: Alpha",alpha,"\t l1_ration =", l1_p,"\tscore=" ,score)
                        max_score = score
                        arr.append(np.array([alpha, l1_p, score]))


        print(arr)
        exit(0)

        timenow = datetime.datetime.now()
        print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp = timenow.strftime("%H%M%S_%d%m%Y")
        f = open('console_output_' + str(stamp) + '.txt', 'w')
        original = sys.stdout
        sys.stdout = Custom_Print(sys.stdout, f)
        opt_wrapper(stamp)


