__author__ = 'eric_rincon'
import numpy

from os import listdir
from pandas import read_csv
from os.path import isfile
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
    Class for loading data from .csv file. Call get_training_test_m first to get data loaded form .csv file.
    To get random samples use def create_random_samples
"""
class DataLoader(object):
    def __init__(self, path='', file = '' ):
        self.dir_path = path
        self.file = file
        if path != '':
            self.csv_files = self.read_csv_files(self.dir_path)

    """
        Returns a new dataset with the over represented portion of the dataset reduced to N number of underrepresented
        label. The over represented label is randomly selected, all of the underrepresented label data is used.
    """
    def underSample(self, features, targets):
        indices_for_all_positives = numpy.asarray(targets.nonzero())[0]
        indices_for_all_negatives = numpy.asarray(numpy.where(targets == 0))[0] # vector indices for the negatives
        positive_element_size = indices_for_all_positives.shape[0]
        random_negative_indices = numpy.random.choice(indices_for_all_negatives, positive_element_size)
        indices = numpy.concatenate([indices_for_all_positives, random_negative_indices])
        numpy.random.shuffle(indices)
        new_targets = targets[indices]
        new_documents = features[indices, :]
        output = "Negative indices: {}\n\n".format(new_documents[:,0])
        return [new_documents, new_targets]

    ""
    def create_random_samples(self, features, targets, train_p=0, valid_p=0, test_p=0, get_ensemble_test_set=False):
        while not(train_p >= 0 or train_p<=1):
            train_p = input('Please enter a value between 0 and 1 for train percentage: ')
        while not(valid_p >= 0 or valid_p<=1):
            valid_p = input('Please enter a value between 0 and 1 for validation percentage: ')
        while not(test_p >= 0 or test_p<=1):
            test_p = input('Please enter a value between 0 and 1 for test percentage: ')

        n_documents = features.shape[0]
        n_indices = n_documents - (n_documents % 10)

        #round to nearest 10 throw away other labels
        features = features[:n_indices,:]
        targets = targets[:n_indices]

        x = [] #features
        y = [] #target labels

        train_set_n_cols = n_documents*train_p
        valid_set_n_cols = n_documents*valid_p + train_set_n_cols
        test_set_n_cols = n_documents*test_p + valid_set_n_cols

        if not(train_p == 0):
            train_features = features[:train_set_n_cols, :]
            train_targets = targets[:train_set_n_cols]
            x.append(train_features)
            y.append(train_targets)

        if not(valid_p == 0):
            valid_features = features[train_set_n_cols: valid_set_n_cols,:]
            valid_targets = targets[train_set_n_cols: valid_set_n_cols]
            x.append(valid_features)
            y.append(valid_targets)
        if not(test_p == 0):
            test_features = features[valid_set_n_cols:test_set_n_cols, :]
            test_targets = targets[valid_set_n_cols:test_set_n_cols]
            x.append(test_features)
            y.append(test_targets)

        if not get_ensemble_test_set:
            return [x, y]
        else:
            x_y = [x, y]
            indices_to_delete = numpy.asarray(range(int(test_set_n_cols+1)))
            features = numpy.delete(features, indices_to_delete, axis=0)
            targets = numpy.delete(targets, indices_to_delete)
            data = [features, targets]
            return [data, x_y]
    """
        columns should be a dictionary that maps the indices to get with their respective names in the csv file.
    """

    def cross_fold_valdation(self, features, targets):
        n = features.shape[0]
        k_fold = KFold(n, 5)
        folds = []
        for train, test in k_fold:
            train_sample = [features[train, :], targets[train]]
            test_sample = [features[test, :], targets[test]]
            output = "Train indices {}\n\nTest indices {}\n\n".format(train, test)
            self.file.write(output)
            fold = [train_sample, test_sample]
            folds.append(fold)
        return folds
    def get_feature_matrix(self, columns):
        #feature_matrix: the return value that will contain all the documents with the appropriate
        #value concatenated to each respective element
        feature_matrix = []

        #True: wil get the feature matrix of the documents
        #False: will return the column requested

        y_i = columns.pop('y')
        keys = list(columns.keys())
        n = 0

        #iterate over all csv files provided
        for csv_file in self.csv_files:
            n_rows = csv_file.shape[0]

            #create a row that represents one document and contains all features
            for row in range(n_rows):
                #document_strings: holds all the elements in the [i,j] portion of the csv file
                #features: used to concatenate all the elements in the selected columns
                #feature_vector: represents a document with all the features
                document_strings = ""
                features = ""
                feature_vector = ""
                #iterates over all the keys in the dictionary and
                for n_key in keys:
                    document_strings = csv_file.iloc[row,n_key].split()

                    n+=1

                    #concatenate the row name to each respective feature
                    for string in document_strings:
                        features += (columns[n_key] + string) + " "

                    feature_vector += features
                feature_matrix.append(numpy.asarray(feature_vector))
        return [numpy.asarray(feature_matrix), numpy.asarray(csv_file.iloc[:, y_i])]
    """
        Main def to call to get data from csv file and outputs in [x_train_tfidf, y].
        x_train_tfidf: is a features matrix that is the n examples by n features. The features are created by
        scikit-learn's tfidf
        y: is vector representing all the corresponding labels labeled 0 or 1.
    """
    def get_training_test_m(self, path, indices, sparse=False):
        data = self.get_feature_matrix(indices)
        feature_matrix = data[0]
        y = data[1]
        count_vect = CountVectorizer()
        x_train_counts = count_vect.fit_transform(feature_matrix)
        tfidf_transformer = TfidfTransformer()
        x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

        if not sparse:
            x_train_tfidf = x_train_tfidf.todense()
        x_train_tfidf = numpy.asarray(x_train_tfidf)

        #Convert y vector into int format and relabel -1 to 0 for Theano
        y = numpy.intc(y)
        y[y == -1] = 0
        n = feature_matrix.shape[0]
        indice_vector = numpy.asarray([range(n)]).T
        x_train_tfidf = numpy.hstack((indice_vector, x_train_tfidf))
        return [x_train_tfidf, y]

    """Takes in a parameter folder, reads all csv files and then returns a list containing the csv files.
    """
    def get_path(self):
        return self.dir_path
    def read_csv_files(self, path):
        csv_file_data = []

        if isfile(path):
            csv_file_data.append(read_csv(path))

            return csv_file_data
        else:
            csv_file_paths = listdir(path)

            for csv_file_path in csv_file_paths:
                if '.csv' in csv_file_path:
                    csv_data = read_csv(csv_file_path)
                    csv_file_data.append(csv_data)

            return csv_file_data

