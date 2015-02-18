__author__ = 'eric_rincon'

import time
import numpy

import theano
import theano.tensor as T

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from MLP import MLP

class NeuralNet:
    """
        Attributes:
            features: Numpy array matrix that represents features
            targets: Numpy array matrix that represents the
    """
    def __init__(self, metric_list="none", parameter_list="none"):
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  #
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        if metric_list == "none":
            self.metrics = {"F1": 0, "Accuracy": 0, "AUC": 0, "Precision": 0, "Recall": 0}
        else:
            self.metrics = metric_list

        if parameter_list == "none":
            self.parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 100, 'batch_size': 10,
                  'n_hidden_units': 1000, 'activation_function': T.tanh, 'n_layers': 1, "train_p": .6}
        else:
            self.parameters = parameter_list
        self.mlp = ""

    def train(self, x_input, y_input):
        """
        Demonstrate stochastic gradient descent optimization for a multilayer
        perceptron

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic gradient

        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)

        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: /datasets/ACEInhibitors_processed.csv

        """

        learning_rate = self.parameters["learning_rate"]
        L1_reg = self.parameters["L1_term"]
        L2_reg = self.parameters["L2_term"]
        n_epochs = self.parameters["n_epochs"]
        batch_size = self.parameters["batch_size"]
        index = T.lscalar('index')  # index to a [mini]batch
        train_size = x_input.shape[0] * self.parameters["train_p"]
        max_size = x_input.shape[0] - (x_input.shape[0] % 10)
        train_set_x = x_input[:train_size, :]
        train_set_y = y_input[:train_size]
        valid_set_x = x_input[(train_size + 1 ):max_size, :]
        valid_set_y = y_input[(train_size + 1):max_size]

        #compute number of minibatches for training, validation and testing
        n_train_batches = int(train_set_x.shape[0] / batch_size)
        n_valid_batches = int(valid_set_x.shape[0] / batch_size)
      #  n_test_batches = int(test_set_x.shape[0] / batch_size)


        number_in = train_set_x.shape[1]

        valid_set_x = theano.shared(valid_set_x, 'valid_set_x')
        valid_set_y = theano.shared(valid_set_y, 'valid_set_y')
        train_set_x = theano.shared(train_set_x, 'train_set_x')
        train_set_y = theano.shared(train_set_y, 'train_set_y')

        # start-snippet-4
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.mlp = MLP(
            rng= numpy.random.RandomState(),
            input=self.x,
            n_in = number_in,
            n_hidden=self.parameters["n_hidden_units"],
            n_out=2,
            a_function = self.parameters["activation_function"]
        )
        cost = (
            self.mlp.negative_log_likelihood(self.y)
            + L1_reg * self.mlp.L1
            + L2_reg * self.mlp.L2_sqr
        )



        # end-snippet-4
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        """
        test_model = theano.function(
            inputs=[index],
            outputs=self.mlp.errors(y),
            givens={
                self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        f1_model = theano.function(
            inputs=[index],
            outputs=self.mlp.f1_score(y),
            givens={
                self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
        """
        validate_model = theano.function(
            inputs=[index],
            outputs=self.mlp.errors(self.y),
            givens={
                self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        # start-snippet-5
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.mlp.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.mlp.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # end-snippet-5

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')

        # early-stopping parameters
        patience = number_in  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        f1_score = 0
        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        """
                        test_losses = [test_model(i) for i
                                       in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        """

                        """
                        f1_scores = [f1_model(i)[0] for i in range(n_test_batches)]
                        f1_score = numpy.mean(f1_scores)

                        precision = [f1_model(i)[1] for i in range(n_test_batches)]
                        precision_avg = numpy.mean(precision)

                        recall = [f1_model(i)[2] for i in range(n_test_batches)]
                        recall_avg = numpy.mean(recall)
                        """
                        """
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                        """

                if patience <= iter:
                    done_looping = True
                    break
                    """
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

              """

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
        test_set_x = theano.shared(x, 'test_set_x')

        W = self.mlp.logRegressionLayer.W
        b = self.mlp.logRegressionLayer.b
        hl_W = self.mlp.hiddenLayer.W
        hl_b = self.mlp.hiddenLayer.b
        input = T.tanh(T.dot(test_set_x, hl_W) + hl_b)

        get_y_pred = theano.function(
            inputs=[],
            outputs=T.argmax(T.nnet.softmax(T.dot(input, W) + b), axis=1),
            on_unused_input='ignore',
        )
        return get_y_pred()
    def __str__(self):
        return "MLP:\nF1 Score Average: {}\nPrecision Average: {}\n" \
                 "Recall Average: {}\nAccuracy: {}\nROC: {}\n".format(self.metrics["F1"],
                                               self.metrics["Precision"],
                                               self.metrics["Recall"],
                                               self.metrics["Accuracy"],
                                              self.metrics["AUC"])
