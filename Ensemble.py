__author__ = 'eric_rincon'

import numpy

from scipy.stats.mstats import mode

from DataLoader import DataLoader
from SVM import SVM
from NeuralNet import NeuralNet

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class Ensemble:
    def __init__(self, metric_list="none", ensemble_size=11, classifier_type="svm",
                 list_of_classifiers=[]):
        self.n_classifiers = ensemble_size
        self.type_of_classifier = classifier_type
        self.classifiers = list_of_classifiers


        if metric_list == "none":
            self.metrics = {"F1": 0, "Accuracy": 0, "AUC": 0, "Precision": 0, "Recall": 0}
        else:
            self.metrics = metric_list
        if list_of_classifiers == []:
            for n in range(self.n_classifiers):
                if classifier_type == "svm":
                    classifier = SVM()
                else:
                    classifier = NeuralNet()
                self.classifiers.append(classifier)
        else:
            self.n_classifiers = len(list_of_classifiers)
    def train(self, x, y):
        data_loader = DataLoader()

        for classifier in self.classifiers:
            sampled_data = data_loader.underSample(x, y)
            x_sampled = sampled_data[0]
            y_sampled = sampled_data[1]
            classifier.train(x_sampled, y_sampled)

    def test(self, x, y):
        assert (not self.classifiers == []), 'There are no classifiers to test. Run train_ensemble first.'
        prediction = self.predict(x)
        f1 = f1_score(y, prediction)
        precision = precision_score(y, prediction)
        recall = recall_score(y, prediction)
        auc = roc_auc_score(y, prediction)
        accuracy = accuracy_score(y, prediction)

        #test classifiers individually
        for classifier in self.classifiers:
            classifier.test(x, y)

        self.metrics["F1"] = f1
        self.metrics["Precision"] = precision
        self.metrics["Recall"] = recall
        self.metrics["AUC"] = auc
        self.metrics["Accuracy"] = accuracy

    def predict(self, x):
        prediction_matrix = numpy.zeros((x.shape[0], self.n_classifiers))
        n = 0

        for classifier in self.classifiers:
            prediction = classifier.predict(x)
            prediction_matrix[:, n] = prediction
            n+=1
        return mode(prediction_matrix, 1)[0].reshape(1, -1)[0]

    def __str__(self):
        output = "Ensemble:\nF1: {}\nPrecision: {}\n" \
                 "Recall: {}\nAccuracy: {}\nAUC: {}\n".format(self.metrics["F1"],
                                               self.metrics["Precision"],
                                               self.metrics["Recall"],
                                               self.metrics["Accuracy"],
                                              self.metrics["AUC"])
        #Iterate over each classifier in the ensemble and output
        #their respective metrics
        for classifier in self.classifiers:
            output+=classifier.__str__()
        return output