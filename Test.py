__author__ = 'eric_rincon'

import numpy
import theano
import os
import sys
from theano import tensor as T
from DataLoader import DataLoader
from Ensemble import Ensemble

#names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
#  values

mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 100, 'batch_size': 20,
                  'n_hidden_units': 1000, 'activation_function': T.tanh, 'n_layers': 1}
#names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
#  values

"""

"""
files = os.listdir("datasets/")
del files[0]
for path in files:
    path = 'datasets/' + path
    file_name = path.split("/")[1].split(".")[0] + ".txt"
    file = open(file_name, 'w+')
    names = {1:'title' ,4:'abstract', 5:'mesh', 'y':6}
    data_loader = DataLoader(path, file)
    data = data_loader.get_training_test_m(path, names)
    features = data[0]
    target_labels = data[1]
    under_samples_features, under_sampled_targets = data_loader.underSample(features, target_labels)
    folds = data_loader.cross_fold_valdation(under_samples_features,
                                    under_sampled_targets)
    k_folds = numpy.array(range(len(folds)))
    for k in k_folds:
        train_sample = folds[k][0]
        train_sample[0] = train_sample[0][:, 1:]
        test_sample = folds[k][1]
        test_sample[0] = test_sample[0][:, 1:]
        output = "Fold {}\n" \
                 "Test set indices {} \n".format(k, test_sample[0][:,1])
        file.write(output)
        ensemble = Ensemble(10, [train_sample, test_sample], file)
        ensemble.train_ensemble(mlp_parameters)
        ensemble.test_ensamble()

    file.close()