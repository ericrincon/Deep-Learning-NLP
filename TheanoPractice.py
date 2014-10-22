__author__ = 'ericRincon'

import theano.tensor as T
import LogisticRegression
import theano

x = T.matrix('x')
y = T.matrix('y')

classifier = LogisticRegression(input = x, n_in = 28, n_out = 10)
cost = classifier.negativeLogLikelihood(y)

#g_W and g_b are symbolic variables which can be used as part of a computation graph
g_W = T.grad(cost = cost, wrt = classifier.W)
g_b = T.grad(cost = cost, wrt = classifier.b)

updates = [((classifier.W, classifier.W) - learning_rate * g_W),
           ((classifier.b, classifier.b) - learning_rate * g_b)]

train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
)

validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
)