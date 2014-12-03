__author__ = 'eric_rincon'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from MLP import MLP
from DataLoader import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def create_random_samples(targets, documents, train_p, valid_p, test_p):
    indices_for_all_positives = numpy.asarray(targets.nonzero())[0]
    indices_for_all_negatives = numpy.asarray(numpy.where(targets == 0))[0] # vector indices for the negatives
    positive_element_size = indices_for_all_positives.shape[0]
    random_negative_indices = numpy.random.choice(indices_for_all_negatives, positive_element_size)
    indices = numpy.concatenate([indices_for_all_positives, random_negative_indices])
    numpy.random.shuffle(indices)
    indices = indices[:360]
    new_targets = targets[indices]
    new_documents = documents[indices, :]
    n_documents = new_documents.shape[0]
    train_set_n_cols = n_documents*train_p

    valid_set_n_cols = n_documents*valid_p + train_set_n_cols
    test_set_n_cols = n_documents*test_p + valid_set_n_cols

    train_features = new_documents[:train_set_n_cols, :]
    train_targets = new_targets[:train_set_n_cols]

    valid_features = new_documents[train_set_n_cols: valid_set_n_cols,:]
    valid_targets = new_targets[train_set_n_cols: valid_set_n_cols]

    test_features = new_documents[valid_set_n_cols:, :]
    test_targets = new_targets[valid_set_n_cols:test_set_n_cols]
    features = [train_features, valid_features, test_features]
    targets = [train_targets, valid_targets, test_targets]

    return [features, targets]

def get_training_test_m(path, indices):
    data_loader = DataLoader(path)
    data = data_loader.get_feature_matrix(indices)
    feature_matrix = data[0]
    y = data[1]
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(feature_matrix)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    return [x_train_tfidf, y]

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=1, n_epochs=1000, batch_size=20, n_hidden=2250):

    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradien

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target values
    path = 'datasets/ACEInhibitors_processed.csv'
    names = {1:'title' ,4:'abstract', 5:'mesh', 'y':6}
    data = get_training_test_m(path, names)
    features = data[0]
    targets = numpy.intc(data[1])
    targets[targets == -1] = 0
    data_sets = numpy.asarray(features.todense())
    train_percent = .6
    valid_percent = .2
    test_percent = .2
    features, targets = create_random_samples(targets, data_sets, train_percent, valid_percent, test_percent)

    train_set_x = features[0]
    train_set_y = targets[0]
    valid_set_x = features[1]
    valid_set_y = targets[1]

    test_set_x = features[2]
    test_set_y = targets[2]
    #compute number of minibatches for training, validation and testing
    print(train_set_y)

    n_train_batches = int(train_set_x.shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.shape[0] / batch_size)
    n_test_batches = int(test_set_x.shape[0] / batch_size)
    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    number_in = test_set_x.shape[1]
    test_set_x = theano.shared(test_set_x, 'test_set_x')
    test_set_y = theano.shared(test_set_y, 'test_set_y')
    valid_set_x = theano.shared(valid_set_x, 'valid_set_x')
    valid_set_y = theano.shared(valid_set_y, 'valid_set_y')
    train_set_x = theano.shared(train_set_x, 'train_set_x')
    train_set_y = theano.shared(train_set_y, 'train_set_y')
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in = number_in,
        n_hidden=n_hidden,
        n_out=2
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


                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)))


if __name__ == '__main__':
    test_mlp()