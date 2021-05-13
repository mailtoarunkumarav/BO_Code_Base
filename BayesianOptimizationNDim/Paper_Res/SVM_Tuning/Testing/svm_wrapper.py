from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

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
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        # Assign colum names to the dataset
        colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        # Read dataset to pandas dataframe
        #irisdata = pd.read_csv(url, names=colnames)
        irisdata = pd.read_csv("../Dataset/iris.data", names=colnames)
        X = irisdata.drop('Class', axis=1)
        y = irisdata['Class']

        # # # # Breast cancer data
        # colnames = [ 'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes','node-caps', 'deg-malig', 'breast','breast-quad','irradiat' ]
        # bcdata = pd.read_csv("Dataset/breast-cancer.data", names=colnames)
        # X = bcdata.drop('Class', axis=1)
        # y = bcdata['Class']


        # # # # Breast cancer wisconsin data
        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        # # bcdata = pd.read_csv("Dataset/wdbc.data")
        # bcdata = pd.read_csv(url)
        # X = bcdata.drop(bcdata.columns[[0,1]], axis=1)
        # y = bcdata.iloc[:,1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # #Random Forests
        # random_forest_classifier = RandomForestClassifier(n_estimators=1000, max_depth=10)
        # random_forest_classifier.fit(X_train, y_train)
        # accuracy = random_forest_classifier.score(X_test, y_test)
        # print(accuracy)
        # exit(0)

        # SV Classifierclf.fit(X_train,y_train)
        # svclassifier = SVC(kernel='linear')
        # svclassifier = SVC(kernel='sigmoid')
        # svclassifier = SVC(kernel='rbf')
        # c=2.32292176
        # g=-2.59608466
        # print(10**c, 10**g)
        # svclassifier = SVC(kernel='rbf', C=(10**c), gamma=(10**g), random_state=42)
        # svclassifier.fit(X_train, y_train)
        # y_pred = svclassifier.predict(X_test)
        # accuracy = svclassifier.score(X_test, y_test)
        # print("sss", accuracy)
        # exit(0)


        # lambda_values = np.logspace(-5,5,11)
        # gamma_values = np.logspace(-5,0,6)
        lambda_values = np.linspace(-5, 5, 100)
        gamma_values = np.linspace(-5, 0, 100)
        # lambda_values = [0.0818546730706902, 0.5400593278542374, 0.25003068519063215 ]
        # gamma_values = [0.016244350109588655, 0.02784941350080644, 0.035087344559027116]

        max_accuracy = -1 * float('inf')
        arr=[]
        for lamb in lambda_values:
            for gamma in gamma_values:
                svclassifier = SVC(kernel='rbf', C=10**lamb, gamma=10**gamma)
                svclassifier.fit(X_train, y_train)
                y_pred = svclassifier.predict(X_test)
                accuracy = svclassifier.score(X_test, y_test)
                if(accuracy > max_accuracy or accuracy == 1):

                    if(accuracy>max_accuracy):
                        print("\nMax: gamma",10**gamma,"\t lambda =", 10**lamb,"\taccuracy=" ,accuracy)
                    max_accuracy = accuracy
                    arr.append(np.array([gamma, lamb, accuracy ]))
                    # print(confusion_matrix(y_test,y_pred))
                    # print("\n\nclassification report ", classification_report(y_test, y_pred))
        print(arr)
        timenow = datetime.datetime.now()
        print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp = timenow.strftime("%H%M%S_%d%m%Y")
        f = open('console_output_' + str(stamp) + '.txt', 'w')
        original = sys.stdout
        sys.stdout = Custom_Print(sys.stdout, f)
        opt_wrapper(stamp)


