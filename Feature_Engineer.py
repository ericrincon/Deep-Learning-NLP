__author__ = 'eric_rincon'


import numpy

from scipy import sparse

class Feature_Engineer():

    """
    train_or_test is set to True be defualt which means that the matrix created will for the training if its False
    its for the test
    """
    def augmented_feature_matrix(self, domains=[], held_out_domain=[], train_or_test=True, n_source_domains=0):
        if train_or_test:
            n_source_domains = len(domains)

        if not train_or_test:
            domain_to_check = held_out_domain[0][0]
        else:
            domain_to_check = domains[0][0]

        if sparse.issparse(domain_to_check):
            h_stack = sparse.hstack
            v_stack = sparse.vstack
        else:
            h_stack = numpy.hstack
            v_stack = numpy.vstack

        if train_or_test:

            for i, data_set in enumerate(domains):
                x, y = data_set

                r = x.shape[0]
                c = x.shape[1]

                temp_matrix = x

                zero_matrix = numpy.zeros((r, c))

                for j in range(n_source_domains):
                    if j == i:
                        temp_matrix = h_stack((temp_matrix, x))
                    else:
                        temp_matrix = h_stack((temp_matrix, zero_matrix))
                if i == 0:
                    augmented_feature_matrix = temp_matrix
                    augmented_y = y
                else:
                    augmented_feature_matrix = v_stack((augmented_feature_matrix, temp_matrix))
                    augmented_y = numpy.hstack((augmented_y, y))
            zero_matrix = numpy.zeros((augmented_feature_matrix.shape[0], c))
            augmented_feature_matrix = h_stack((augmented_feature_matrix, zero_matrix))

        if not len(held_out_domain) == 0:
            held_out_x, held_out_y = held_out_domain
            held_out_x_copy = held_out_x

            if train_or_test:
                n_features = augmented_feature_matrix.shape[1]-2*c
            else:
                n_features = n_source_domains*held_out_x.shape[1]
                augmented_feature_matrix = held_out_x
            zero_matrix = numpy.zeros((held_out_x.shape[0], n_features))
            held_out_x = h_stack((held_out_x, zero_matrix))
            held_out_x = h_stack((held_out_x, held_out_x_copy))

            if train_or_test:
                augmented_y = numpy.hstack((augmented_y, held_out_y))
                augmented_feature_matrix = v_stack((augmented_feature_matrix, held_out_x))
            else:
                augmented_feature_matrix = held_out_x
                augmented_y = held_out_y
        return [augmented_feature_matrix, augmented_y]