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
from Domain_MLP import Domain_MLP

from MLP import MLP

class NeuralNet:
    """
        Attributes:
            features: Numpy array matrix that represents features
            targets: Numpy array matrix that represents the
    """
    def __init__(self, n_hidden_units, batch_size, output_size, metric_list="none", learning_rate=1, l1_term=0,
                 l2_term=0, n_epochs=100, activation_function='tanh', train_p=.6, dropout=False, dropout_rate=.5,
                 momentum=False, momentum_term=.9, adaptive_learning_rate=False):
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  #
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if metric_list == "none":
            self.metrics = {"F1": 0, "Accuracy": 0, "AUC": 0, "Precision": 0, "Recall": 0}
        else:
            self.metrics = metric_list

        self.learning_rate = learning_rate
        self.L1_reg = l1_term
        self.L2_reg = l2_term
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_percent = train_p

        #Define new ReLU activation function
        def relu(x):
            return T.switch(x < 0, 0, x)

        if activation_function == 'relu':
            self.activation_function = relu
        elif activation_function == 'tanh':
            self.activation_function = T.tanh
        elif activation_function == 'sigmoid':
            self.activation_function = T.nnet.sigmoid

        self.output_size = output_size
        self.hidden_layer_sizes = n_hidden_units
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.momentum_term = momentum_term


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


        index = T.lscalar('index')  # index to a [mini]batch
        train_size = x_input.shape[0] * self.train_percent
        max_size = x_input.shape[0] - (x_input.shape[0] % 10)
        train_set_x = x_input[:train_size, :]
        train_set_y = y_input[:train_size]
        valid_set_x = x_input[(train_size + 1 ):max_size, :]
        valid_set_y = y_input[(train_size + 1):max_size]

        #compute number of minibatches for training, validation and testing
        n_train_batches = int(train_set_x.shape[0] / self.batch_size)
        n_valid_batches = int(valid_set_x.shape[0] / self.batch_size)
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
            n_out=self.output_size,
            a_function = self.activation_function,
            n_hidden_sizes=self.hidden_layer_sizes,
            dropout=self.dropout,
            dropout_rate=self.dropout_rate
        )

        cost = (
            self.mlp.negative_log_likelihood(self.y)
            + self.L1_reg * self.mlp.L1
            + self.L2_reg * self.mlp.L2_sqr
        )

        # end-snippet-4
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch

        validate_model = theano.function(
            inputs=[index],
            outputs=self.mlp.errors(self.y),
            givens={
                self.x: valid_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: valid_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        training_errors = theano.function(
            inputs=[index],
            outputs=self.mlp.errors(self.y),
            givens={
                self.x: train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        # start-snippet-5
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        parameter_gradients = [T.grad(cost, param) for param in self.mlp.params]


        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        updates = []

        if self.momentum:
            delta_before=[]
            for param_i in self.mlp.params:
                delta_before_i=theano.shared(value=numpy.zeros(param_i.get_value().shape))
                delta_before.append(delta_before_i)

            for param, parameter_gradients, delta_before_i in zip(self.mlp.params, parameter_gradients, delta_before):
                delta_i = -self.learning_rate * parameter_gradients + self.momentum_term*delta_before_i

                updates.append((param, param + delta_i))
                updates.append((delta_before_i,delta_i))
        else:
            for param, parameter_gradients in zip(self.mlp.params, parameter_gradients):
                updates.append((param, param - self.learning_rate * parameter_gradients))



        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
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
        start_time = time.clock()
        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    training_losses = [training_errors(i) for i
                                        in range(n_train_batches)]
                    this_training_loss = numpy.mean(training_losses)
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    output = "epoch {}, minibatch {}/{}, training error {}, validation error {}".format(
                        epoch,
                        (minibatch_index + 1),
                        n_train_batches,
                        this_training_loss,
                        this_validation_loss
                    )
                    print(output)
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

    def setup_labels(self, y):

        assert "There is no need to relabel if n_classes < 2 ", y < 2

        negative_example_label = 2

        #Transform matrices and relabel them for the neural network
        for i, yi in enumerate(y):
            if i > 0:
                negative_example_label = negative_example_label+2
            positive_example_label = negative_example_label+1

            relabeled_y = yi
            relabeled_y[relabeled_y == 0] = negative_example_label
            relabeled_y[relabeled_y == 1] = positive_example_label

            if i == 0:
                neural_net_y = relabeled_y
            else:
                neural_net_y = numpy.hstack((neural_net_y, relabeled_y))
        neural_net_y = numpy.intc(neural_net_y)
        return neural_net_y

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
        #Create a theano shared variable for the input x: the data to be predicted
        test_set_x = theano.shared(x, 'test_set_x')
        input = test_set_x


        #Iterate over all the hidden layers in the MLP
        for i_hidden_layer, hidden_layer in enumerate(self.mlp.hidden_layers):
            hl_W = hidden_layer.W
            hl_b = hidden_layer.b

            weight_matrix = self.activation_function(T.dot(input, hl_W) + hl_b)

            #Multiply the weights by the expected value of the dropout which is just the
            #dropoutrate so in most cases half the weights but only at test time
            if self.dropout:
                weight_matrix *= self.dropout_rate
            input = weight_matrix

        #Get the weights and bias from the softmax output layer
        W = self.mlp.logRegressionLayer.W
        b = self.mlp.logRegressionLayer.b

        #compile the thenao function for calculating the outputs from the softmax layer
        get_y_prediction = theano.function(
            inputs=[],
            outputs=T.argmax(T.nnet.softmax(T.dot(weight_matrix, W) + b), axis=1),
            on_unused_input='ignore',
        )
        return get_y_prediction()

    def __str__(self):
        return "MLP:\nF1 Score: {}\nPrecision: {}\n" \
                 "Recall: {}\nAccuracy: {}\nROC: {}\n".format(self.metrics['F1'],
                                               self.metrics['Precision'],
                                               self.metrics['Recall'],
                                               self.metrics['Accuracy'],
                                              self.metrics['AUC'])

