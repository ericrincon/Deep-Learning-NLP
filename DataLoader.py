__author__ = 'eric_rincon'
import numpy

from os import listdir
from pandas import read_csv
from os.path import isfile

from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import scipy.sparse

"""
    Class for loading data from .csv file. Call get_training_test_m first to get data loaded form .csv file.
    To get random samples use def create_random_samples
"""
class DataLoader(object):
    def __init__(self, input_data=''):
        self.data = input_data
        self.count_vect = CountVectorizer(stop_words={"english"}, ngram_range=(1, 2), max_features=10000)
        self.tfidf_transformer = TfidfTransformer()

        if not input_data == '':
            self.csv_files = self.read_csv_files(self.data)

    """
        Returns a new dataset with the over represented portion of the dataset reduced to N number of underrepresented
        label. The over represented label is randomly selected, all of the underrepresented label data is used.
    """
    def underSample(self, features, targets):

        indices_for_all_positives = numpy.asarray(targets.nonzero())[0]
        #Vector indices for the negatives
        indices_for_all_negatives = numpy.asarray(numpy.where(targets == 0))[0]
        positive_element_size = indices_for_all_positives.shape[0]
        random_negative_indices = numpy.random.choice(indices_for_all_negatives, positive_element_size)
        indices = numpy.concatenate([indices_for_all_positives, random_negative_indices])

        numpy.random.shuffle(indices)
        new_targets = targets[indices]
        new_documents = features[indices, :]
        output = "Negative indices: {}\n\n".format(new_documents[:,0])
        return [new_documents, new_targets]

    """
        Returns a train and test sample with features of the data in the x list and the labels in the y list
        ordered by indices
    """

    def shuffle_x_and_y(self, x, y):
        n = x.shape[0]
        indices = numpy.arange(n)
        numpy.random.shuffle(indices)
        if len(x.shape) == 2:
            x = x[indices, :]
            y = y[indices]
        else:
            x = x[indices]
            y = y[indices]
        return [x, y]
    def create_random_samples(self, features, targets, train_p=0, valid_p=0, test_p=0, get_ensemble_test_set=False):
        while not(train_p >= 0 or train_p<=1):
            train_p = input('Please enter a value between 0 and 1 for train percentage: ')
        while not(valid_p >= 0 or valid_p<=1):
            valid_p = input('Please enter a value between 0 and 1 for validation percentage: ')
        while not(test_p >= 0 or test_p<=1):
            test_p = input('Please enter a value between 0 and 1 for test percentage: ')
        n_documents = features.shape[0]
        limit = features.shape[0] - (n_documents % 10)
        data = self.shuffle_x_and_y(features, targets)
        features = data[0][:limit, :]
        targets = data[1][:limit]
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

    """
        Returns n_folds folds (default set to five) with each fold consisting of
        a train sample and test sample. Within each train/test sample there is a list
        object containing the x and y
    """
    def cross_fold_valdation(self, features, targets, n_folds=5):
        n = features.shape[0]
        k_fold = KFold(n, n_folds, shuffle=True)
        folds = []

        for train, test in k_fold:
            if len(features.shape) > 1:
                train_sample = [features[train, :], targets[train]]
                test_sample = [features[test, :], targets[test]]
            else:
                train_sample = [features[train], targets[train]]
                test_sample = [features[test], targets[test]]
            fold = [train_sample, test_sample]
            folds.append(fold)
        return folds
    def get_feature_matrix(self, columns, data="", path=""):
        #feature_matrix: the return value that will contain all the documents with the appropriate
        #value concatenated to each respective element

        if isinstance(data, type("")):
            data = self.csv_files
        else:
            data = [data]
        feature_matrix = []
        targets = []
        #True: wil get the feature matrix of the documents
        #False: will return the column requested
        y_i = columns.pop('y')
        keys = list(columns.keys())
        n = 0

        #iterate over all csv files provided
        for i, csv_file in enumerate(data):
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

            y_to_append = numpy.asarray(csv_file.iloc[:, y_i])
            if i > 0:
                targets = numpy.concatenate((targets, y_to_append))
            else:
                targets = y_to_append
        targets = numpy.asarray(targets).flatten()
        columns.update({'y': 6})

        #Convert y vector into int format and relabel -1 to 0 for Theano
        targets = numpy.asarray(targets)
        targets = numpy.intc(targets)
        targets[targets == -1] = 0

        return [numpy.asarray(feature_matrix), targets]
   #     return [numpy.asarray(feature_matrix), numpy.asarray(csv_file.iloc[:, y_i])]
    """
        Main def to call to get data from csv file and outputs in [x_train_tfidf, y].
        x_train_tfidf: is a features matrix that is the n examples by n features. The features are created by
        scikit-learn's tfidf
        y: is vector representing all the corresponding labels labeled 0 or 1.
    """
    def get_transformed_features(self, x, sparse=False, add_index_vector=True):
        """
        if isinstance(data, type("")):
            data = self.get_feature_matrix(indices)
        else:
            data = self.get_feature_matrix(data=[data], columns=indices)
        """
        feature_matrix = x
        x_train_counts = self.count_vect.fit_transform(feature_matrix)
        x_train_tfidf = self.tfidf_transformer.fit_transform(x_train_counts)
        if not sparse:
            x_train_tfidf = x_train_tfidf.todense()

        if add_index_vector:
            n = feature_matrix.shape[0]
            index_vector = numpy.asarray([range(n)]).T

            if not sparse:
                x_train_tfidf = numpy.hstack((index_vector, x_train_tfidf))
            else:
                x_train_tfidf = scipy.sparse.hstack((index_vector, x_train_tfidf))
        return x_train_tfidf
    def transform(self, x, sparse=False):
        counts = self.count_vect.transform(x)
        tfidf = self.tfidf_transformer.transform(counts)

        if sparse:
            return tfidf
        else:
            return tfidf.todense()
    """Takes in a parameter folder, reads all csv files and then returns a list containing the csv files.
    """
    def get_path(self):
        return self.dir_path
    def read_csv_files(self, input_data, read_all=False):
        csv_file_data = []

        if isinstance(input_data, list):
            for csv_file_path in input_data:
                if '.csv' in csv_file_path:
                    csv_data = read_csv(csv_file_path)
                    csv_file_data.append(csv_data)
        elif isfile(input_data) and not read_all:
            csv_file_data.append(read_csv(input_data))

            return csv_file_data
        else:
            csv_file_paths = listdir(input_data)
            for csv_file_path in csv_file_paths:
                if '.csv' in csv_file_path:
                    csv_data = read_csv(input_data + csv_file_path)
                    csv_file_data.append(csv_data)

            return csv_file_data

