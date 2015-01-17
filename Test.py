__author__ = 'eric_rincon'

import numpy
import theano

from theano import tensor as T
from Ensemble import Ensemble

#names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
#  values

mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .01, 'n_epochs': 1, 'batch_size': 10,
                  'n_hidden_units': 1000, 'activation_function': T.tanh, 'n_layers': 1}
#names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
#  values

ensemble = Ensemble(10)
ensemble.train_ensemble(mlp_parameters)
ensemble.test_ensamble()

