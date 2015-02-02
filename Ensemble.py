__author__ = 'eric_rincon'

import theano
import numpy
import pickle

import sys

from scipy.stats.mstats import mode
from sklearn.metrics import roc_auc_score
from DataLoader import DataLoader
from RunMLP import RunMLP
from MLP import MLP
from theano import tensor as T

class Ensemble:
    def __init__(self, n, data, file = ''):
        self.n_classifiers = n
        self.classifiers = []
        self.training_data = data[0]
        self.test_data = data[1]
        self.file = file


    def train_ensemble(self, mlp_parameters):
        features = self.training_data[0]
        target_labels = self.training_data[1]
        train_percent = .6
        valid_percent = .2
        test_percent = .2
        test_features = self.test_data[0]
        test_target_labels = self.test_data[1]
        for n in range(self.n_classifiers):
            output = "Classifier {} ".format(n)
            self.file.write(output)
            data_loader = DataLoader()
            sampled_data = data_loader.create_random_samples(features=features, targets=target_labels, train_p=.6,valid_p=.2)
            x = sampled_data[0]
            x.append(test_features)
            y = sampled_data[1]
            y.append(test_target_labels)
            mlp = RunMLP(x, y, n_layers=mlp_parameters['n_layers'])
            print("Classifier ", n)
            classifier = mlp.run(learning_rate=mlp_parameters['learning_rate'], L1_reg=mlp_parameters['L1_term'],
                                 L2_reg=mlp_parameters['L2_term'], n_epochs=mlp_parameters['n_epochs'],
                                 batch_size=mlp_parameters['batch_size'],
                                 n_hidden_units=mlp_parameters['n_hidden_units'],
                                 activation_function=mlp_parameters['activation_function'], file=self.file)
            self.classifiers.append(classifier)
    def test_ensamble(self, data=[]):
        if not data:
            data = self.test_data

        test_set_x = theano.shared(data[0], 'test_set_x')
        test_set_y = data[1]

        assert (not self.classifiers == []), 'There are no classifiers to test. Run train_ensemble first.'
        predictions = []

        for classifier in self.classifiers:
            W = classifier.logRegressionLayer.W
            b = classifier.logRegressionLayer.b
            hl_W = classifier.hiddenLayer.W
            hl_b = classifier.hiddenLayer.b
            input = T.tanh(T.dot(test_set_x, hl_W) + hl_b)

            get_y_pred = theano.function(
                inputs=[],
                outputs=T.argmax(T.nnet.softmax(T.dot(input, W) + b), axis=1),
                on_unused_input='ignore',
            )
            predictions.append(get_y_pred())
        prediction_matrix = predictions.pop(0)[:, numpy.newaxis]
        #Work in progress
        for prediction in predictions:
            auc = roc_auc_score(test_set_y, prediction)
            print('ROC score: ', auc)
            prediction_matrix = numpy.hstack((prediction_matrix, prediction[:, numpy.newaxis]))
        predictions = mode(prediction_matrix, 1)
        n_correct = numpy.sum(numpy.equal(predictions, test_set_y[:, numpy.newaxis])[0])
        percentage_correct = n_correct/test_set_y.shape[0]
        output = 'Ensemble percentage correct: {}%\n'.format(percentage_correct*100)
        self.file.write(output)
        print(output)
