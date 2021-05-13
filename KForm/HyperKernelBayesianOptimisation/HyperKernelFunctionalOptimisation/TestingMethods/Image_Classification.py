import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Test:

    def run_utility(self):

        image_size = 28
        no_of_different_labels = 10
        image_pixels = image_size * image_size
        data_path = ""
        train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
        test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

        fac = 0.99 / 255
        train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
        test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

        train_labels_original = np.asfarray(train_data[:, :1])
        test_labels_original = np.asfarray(test_data[:, :1])

        train_labels_new_index = np.where(train_labels_original == 1 )
        print(train_labels_new_index[0])
        print(train_labels_original[train_labels_new_index[0][1]])
        train_labels_new_array = [train_labels_original[i] for i in train_labels_new_index[0]]

        print(train_labels_new_array)
        exit(0)




        # test_labels_new = np.where[test_labels_original == 1 or test_labels_original == 7]



        # for each_label_train in train_labels_original:



        lr = np.arange(10)

        for label in range(10):
            one_hot = (lr == label).astype(np.int)
            print("label: ", label, " in one-hot representation: ", one_hot)

        lr = np.arange(no_of_different_labels)

        # transform labels into one hot representation
        train_labels_one_hot = (lr == train_labels_original).astype(np.float)
        test_labels_one_hot = (lr == test_labels_original).astype(np.float)

        # we don't want zeroes and ones in the labels neither:
        train_labels_one_hot[train_labels_one_hot == 0] = 0.01
        train_labels_one_hot[train_labels_one_hot == 1] = 0.99
        test_labels_one_hot[test_labels_one_hot == 0] = 0.01
        test_labels_one_hot[test_labels_one_hot == 1] = 0.99

        for i in range(10):
            img = train_imgs[i].reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()




if __name__ == "__main__":
    test_obj = Test()
    test_obj.run_utility()



