__author__ = 'eric_rincon'

from os import listdir
from pandas import read_csv
from os.path import isfile
from array import array
from sys import exit
from pandas import DataFrame as df
import numpy
import sys


class DataLoader(object):
    def __init__(self, path):
        self.dir_path = path
        self.csv_files = self.read_csv_files(self.dir_path)
    """
        columns_to_get should be a dictionary that maps the indices to get with their respective names
    """
    def get_feature_matrix(self, columns):
        #feature_matrix: the return value that will contain all the documents with the appropriate
        #value concatenated to each respective element
        feature_matrix = []

        #True: wil get the feature matrix of the documents
        #False: will return the column requested

        y_i = columns.pop('y')
        keys = list(columns.keys())

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
                    #concatenate the row name to each respective feature
                    for string in document_strings:
                        features += (columns[n_key] + string) + " "
                    feature_vector+=features
                    feature_matrix.append(numpy.asarray(feature_vector))

        return [numpy.asarray(feature_matrix), numpy.asarray(csv_file.iloc[:, y_i])]

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

