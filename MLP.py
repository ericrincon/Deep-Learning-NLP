__author__ = 'eric_rincon'


import theano.tensor as T
from theano import shared
from theano import function
from LogisticRegression import LogisticRegression
from HiddenLayer import HiddenLayer

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden_sizes, n_out, a_function=T.tanh):
        self.rng=rng,
        self.hidden_layer_sizes = n_hidden_sizes
        self.input=input,
        self.n_in=n_in,
        #get the size of the last hidden layer this is the size of the input for softmax output layer
        self.n_out=n_hidden_sizes[len(n_hidden_sizes)-1],
        self.hidden_layers = []
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden_sizes: list
        :param n_hidden: list of number of hidden units at each hidden layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function


        #Create hidden layers for the mlp

        for i_hidden_layer, i_hidden_layer_size in enumerate(n_hidden_sizes):
            if i_hidden_layer > 0:
                n_in = n_hidden_sizes[i_hidden_layer-1]
                input = self.hidden_layers[i_hidden_layer - 1].output

            hidden_layer = HiddenLayer(rng=rng,
                                        input=input,
                                        n_in=n_in,
                                        n_out=i_hidden_layer_size,
                                        activation=a_function)
            self.hidden_layers.append(hidden_layer)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer

        self.logRegressionLayer = LogisticRegression(
            input=self.hidden_layers[len(self.hidden_layers) - 1].output,
            n_in=n_hidden_sizes[len(n_hidden_sizes) - 1],
            n_out=n_out
        )
                # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        l1_weight = shared(0)
        for hl in self.hidden_layers:
            l1_weight += abs(hl.W).sum()
        self.L1 = (
            l1_weight + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        l2_weight = shared(0)

        for hl in self.hidden_layers:
            l2_weight += (hl.W ** 2).sum()

        self.L2_sqr = (
            l2_weight + (self.logRegressionLayer.W ** 2).sum()
        )
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        #Returns the list [f1_score, precision, recall]
        self.f1_score = self.logRegressionLayer.f1_score

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        hidden_layer_params = []

        for hl in self.hidden_layers:
            hidden_layer_params += hl.params

        self.params = hidden_layer_params + self.logRegressionLayer.params
          # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
