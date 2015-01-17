__author__ = 'eric_rincon'

import time

import numpy

import theano
import theano.tensor as T
import os
import sys

from MLP import MLP

class RunMLP(object):

    """
        Attributes:
            features: Numpy array matrix that represents features
            targets: Numpy array matrix that represents the
    """
    def __init__(self, f, t, n_layers=1):
        self.features = f
        self.targets = t
        self.number_layers = n_layers

    def run(self, learning_rate=0.01, L1_reg=0.00, L2_reg=.01, n_epochs=5, batch_size=20, n_hidden_units=100,
            activation_function = T.tanh):

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

        train_set_x = self.features[0]
        train_set_y = self.targets[0]
        valid_set_x = self.features[1]
        valid_set_y = self.targets[1]
        test_set_x = self.features[2]
        test_set_y = self.targets[2]

        #compute number of minibatches for training, validation and testing

        n_train_batches = int(train_set_x.shape[0] / batch_size)
        n_valid_batches = int(valid_set_x.shape[0] / batch_size)
        n_test_batches = int(test_set_x.shape[0] / batch_size)

        # allocate symbolic variables for the data
        index = T.lscalar('index')  # index to a [mini]batch
        x = T.matrix('x')  #
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        number_in = test_set_x.shape[1]
        test_set_x = theano.shared(test_set_x, 'test_set_x')
        test_set_y = theano.shared(test_set_y, 'test_set_y')
        valid_set_x = theano.shared(valid_set_x, 'valid_set_x')
        valid_set_y = theano.shared(valid_set_y, 'valid_set_y')
        train_set_x = theano.shared(train_set_x, 'train_set_x')
        train_set_y = theano.shared(train_set_y, 'train_set_y')
        rng = numpy.random.RandomState()
        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in = number_in,
            n_hidden=n_hidden_units,
            n_out=2,
            a_function = activation_function
        )

        # start-snippet-4
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
        # end-snippet-4
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch

        test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        f1_model = theano.function(
            inputs=[index],
            outputs=classifier.f1_score(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        # start-snippet-5
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # end-snippet-5

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
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
                        test_losses = [test_model(i) for i
                                       in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        f1_scores = [f1_model(i) for i in range(n_test_batches)]
                        f1_score = numpy.mean(f1_scores)




                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))


                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()

        print("f1 score ", f1_score)
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.)))

        return classifier