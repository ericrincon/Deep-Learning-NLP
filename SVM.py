__author__ = "eric_rincon"

import numpy

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class SVM:

    def __init__(self, metric_list="none", parameter_list="none"):
        if metric_list == "none":
            self.metrics = {"F1": 0, "Accuracy": 0, "AUC": 0, "Precision": 0, "Recall": 0}
        else:
            self.metrics = metric_list

        if parameter_list == "none":
            c_range = 10. ** numpy.arange(-2, 3)
            gamma_range = 10. ** numpy.arange(-3, 1)
            self.parameters = {"kernel": "rbf", "c_range": c_range, "gamma_range": gamma_range, "score_model":"f1"}
        else:
            self.parameters = parameter_list
        svm_classifier = svm.SVC(kernel=self.parameters["kernel"])
        self.classifier = GridSearchCV(svm_classifier, scoring=self.parameters["score_model"], param_grid=dict(
                        C=self.parameters['c_range'], gamma=self.parameters["gamma_range"]),n_jobs=-1)

    def train(self, x, y):
        self.classifier.fit(x, y)
    def test(self, x, y):
        prediction = self.predict(x)
        f1 = f1_score(y, prediction)
        precision = precision_score(y, prediction)
        recall = recall_score(y, prediction)
        auc = roc_auc_score(y, prediction)
        accuracy = accuracy_score(y, prediction)

        self.metrics["F1"] = f1
        self.metrics["Precision"] = precision
        self.metrics["Recall"] = recall
        self.metrics["AUC"] = auc
        self.metrics["Accuracy"] = accuracy

    def predict(self, x):
        return self.classifier.predict(x)
    def __str__(self):
        return "SVM:\nF1 Score Average: {}\nPrecision Average: {}\n" \
                 "Recall Average: {}\nError: {}\nROC: {}\nHyperParameters: {}\n".format(self.metrics["F1"],
                                               self.metrics["Precision"],
                                               self.metrics["Recall"],
                                               self.metrics["Accuracy"],
                                              self.metrics["AUC"],
                                              self.parameters)
