__author__ = 'eric_rincon'

import theano
import numpy
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        #initialize with 0, constructs a matrix of size n_in, n_out
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True #It is a safe practice (and a good idea) to use borrow = True, http://deeplearning.net/software/theano/tutorial/aliasing.html
        )

        #initalize the biases b as a vector of n_out
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        #symbolic expression for computing the matrix of class-membership probabilities
        #W is a matrix where column-k represents the separation for class-k
        #x is a matrix where row-j represents input training sample-j
        #b is a vector where element-k represents the free parameter of hyper plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        #Symbolic description of how to computer prediction as class whose probablity is maxmial
        #To get the actual model prediction, we can use the T.argmax operator, which will return the index
        #at which p_y_given_x is maximal (i.e. the class with maximum probability).
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0], y)])
    def errors(self, y):
        if y.ndim != self.p_y_given_x:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neg(self.y_pred, y))
        else:
            NotImplementedError()
