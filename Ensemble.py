__author__ = 'eric_rincon'

import theano
import numpy
from scipy.stats.mstats import mode

from DataLoader import DataLoader
from RunMLP import RunMLP
from MLP import MLP
from theano import tensor as T
class Ensemble:
    def __init__(self, n):
        self.n_classifiers = n
        self.classifiers = []
        self.test_x_y = []


    def train_ensemble(self, mlp_parameters):
        path = 'datasets/ACEInhibitors_processed.csv'
        names = {1:'title' ,4:'abstract', 5:'mesh', 'y':6}
        data_loader = DataLoader(path)
        data = data_loader.get_training_test_m(path, names)
        features = data[0]
        target_labels = data[1]
        train_percent = .6
        valid_percent = .2
        test_percent = .2
        updated_data, test_data  = data_loader.create_random_samples(features, target_labels, test_p=test_percent,
                                                                     get_ensemble_targets=True)
        features = updated_data[0]
        target_labels = updated_data[1]
        test_features = test_data[0][0]
        test_target_labels = test_data[1][0]
        self.test_x_y.append(test_features)
        self.test_x_y.append(test_target_labels)

        for n in range(self.n_classifiers):
            sampled_data = data_loader.create_random_samples(features, target_labels, train_p=.6,valid_p=.2)
            x = sampled_data[0]
            x.append(test_features)
            y = sampled_data[1]
            y.append(test_target_labels)
            mlp = RunMLP(x, y, n_layers=mlp_parameters['n_layers'])
            classifier = mlp.run(learning_rate=mlp_parameters['learning_rate'], L1_reg=mlp_parameters['L1_term'],
                                 L2_reg=mlp_parameters['L2_term'], n_epochs=mlp_parameters['n_epochs'],
                                 batch_size=mlp_parameters['batch_size'],
                                 n_hidden_units=mlp_parameters['n_hidden_units'],
                                 activation_function=mlp_parameters['activation_function'])
            self.classifiers.append(classifier)

    def test_ensamble(self, data=[]):
        if not data:
            data = self.test_x_y

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

        for prediction in predictions:
            prediction_matrix = numpy.hstack((prediction_matrix, prediction[:, numpy.newaxis]))
        predictions = mode(prediction_matrix, 1)
        n_correct = numpy.sum(numpy.equal(predictions, test_set_y[:, numpy.newaxis])[0])
        percentage_correct = n_correct/test_set_y.shape[0]
        print('Ensemble percentage correct: ', percentage_correct*100, '%')
