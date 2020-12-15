import numpy as np
import csv
import math
from tabulate import tabulate
from random import shuffle


class LogisticRegressionClassifier:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.dataset_size = None
        self.dataset = None
        self.feature_size = None
        self.weights = None

    def read_file(self, filename, feature_size):
        self.feature_size = feature_size
        reader = list(csv.reader(open(filename, "rb"), delimiter=" "))
        self.dataset_size = len(reader)
        self.dataset =  np.array(list(map(lambda line: [np.array(line[0:feature_size]).astype("float"), float(line[feature_size])], reader)))
        return self

    # calculating based on https://ivorix.com/time-series-analysis/normalization-to-zero-mean-and-unit-standard-deviation/
    def normalize_data(self):
        for j in range(0, self.feature_size):
            mean = 0
            standard_deviation = 0
            for i in range(0, self.dataset.shape[0]):
                mean += self.dataset[:, 0][i][j]
            mean /= self.dataset.shape[0]
            for i in range(0, self.dataset.shape[0]):
                standard_deviation += (self.dataset[:, 0][i][j] - mean) ** 2
            standard_deviation = math.sqrt(standard_deviation / (self.dataset.shape[0] - 1))
            for i in range(0, self.dataset.shape[0]):
                self.dataset[:, 0][i][j] = (self.dataset[:, 0][i][j] - mean) / standard_deviation
        return self

    def train_test_splitter(self, train_ratio=0.8):
        if train_ratio>0.9:
            raise Exception('''Sorry, you are setting very small ratio for test set and it's not acceptable
                            please choose a train_ratio less than or equal 0.9''')
        shuffle(self.dataset)
        splitting_index = int(len(self.dataset) * train_ratio)
        self.train_set, self.test_set = self.dataset[:splitting_index], self.dataset[splitting_index:]
        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp((-z)))

    def cost_function(self, X, y, w, lambda_reg):
        z = np.dot(X, w)
        epsilon = 1e-5
        return (-y * np.log(self.sigmoid(z) + epsilon) - (1 - y) * np.log(1 - self.sigmoid(z) + epsilon)).mean() + (
                    (np.dot(w, w) * lambda_reg) / (2 * len(X)))

    def train(self, iterations, learning_rate, lambda_reg):
        bayas_raw = np.array([1 for i in range(self.train_set[:, 0].shape[0])]).astype("float").reshape(-1, 1)
        array_X = np.stack(list(self.train_set[:, 0]), axis=0)
        array_X_bayas =np.append(array_X, bayas_raw, axis=1)
        array_y = self.train_set[:, 1]
        loss = []
        weights = np.random.rand(array_X_bayas.shape[1])
        for iteration in range(iterations):
            y_hat = self.sigmoid(np.dot(array_X_bayas, weights))
            derivatives = ( np.dot(array_X_bayas.T, y_hat-array_y)/len(array_X_bayas) ) + (np.dot(weights,lambda_reg)/len(array_X_bayas))
            weights -= np.dot(derivatives,learning_rate).astype("float")
            loss.append(self.cost_function(array_X_bayas, array_y, weights, lambda_reg))
        self.weights = weights
        return self


    def batch_predict(self, X, w):
        z = np.dot(X, w)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

    def batch_detailed_accuracy(self, y, y_pre):
        false_negative, false_positive, true_positive, true_negative = (0,0,0,0)
        for i in range(y.shape[0]):
            if y_pre[i] == 1 :
                if y[i] == 1 :
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if y[i] == 1 :
                    false_negative += 1
                else:
                    true_negative += 1
        return false_negative, false_positive, true_positive, true_negative


    def accuracy (self):
        bayas_raw = np.array([1 for i in range(self.test_set[:, 0].shape[0])]).astype("float").reshape(-1, 1)
        array_X = np.stack(list(self.test_set[:, 0]), axis=0)
        array_X_bayas = np.append(array_X, bayas_raw, axis=1)
        array_y = np.array(list(map(lambda record: int(record), self.test_set[:, 1])))
        array_predicted_y = self.batch_predict(array_X_bayas, self.weights)
        false_negative, false_positive, true_positive, true_negative = self.batch_detailed_accuracy(array_y, array_predicted_y)
        # print(false_negative, false_positive, true_positive, true_negative)
        precision = float(true_positive) / float(true_positive + false_positive)
        recall =  float(true_positive) / float(false_negative + true_positive)
        accuracy = (float(true_positive) + float(true_negative))/len(self.test_set)
        table = [
                    ["Precision", precision],
                    ["Recall", recall],
                    ["Accuracy", accuracy]
                ]
        print(tabulate(table))
        return accuracy

class LogisticRegressionClassifierCrossValidation(LogisticRegressionClassifier):
    def __init__(self, number_of_blocks):
        self.number_of_blocks = number_of_blocks
        self.accuracy_list = []
        self.blocks = []

    def make_blocks(self):
        shuffle(self.dataset)
        self.blocks = np.array_split(self.dataset, self.number_of_blocks)
        return self

    def split_train(self, iterations, learning_rate, lambda_reg, train_ratio=0.8):
        for index, block in enumerate(self.blocks):
            self.dataset = block
            self.train_test_splitter(train_ratio)
            accuracy = self.train(iterations, learning_rate, lambda_reg).accuracy()
            self.accuracy_list.append(accuracy)
            print("accuracy in iteration " + str(index+1) + " : " + str(accuracy))
        print("final Accuracy is :  " + str(np.array(self.accuracy_list).mean()))

# lg = LogisticRegressionClassifier()
# # Read Dataset from file and normalized the data
# lg.read_file("../Datasets/spam.data", 57).normalize_data().train_test_splitter(0.8).train(1000, 0.25, 0.15).accuracy()
#

lg = LogisticRegressionClassifierCrossValidation(10)
lg.read_file("../Datasets/spam.data", 57).normalize_data()\
    .make_blocks().split_train(1000, 0.25, 0.15, 0.8)