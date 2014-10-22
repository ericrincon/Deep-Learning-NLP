__author__ = 'eric_rincon'

from os import listdir
from pandas import read_csv
from os.path import isfile
import sys
class DataLoader(object):
    def __init__(self, path):
        self.dir_path = path
        self.csv_files = self.read_csv_files(self.dir_path)

    def read_column_as_list(self, columns_to_get):
        matrix = []

        for csv_file in self.csv_files:
            matrix.append(csv_file.iloc[:,columns_to_get])
        return matrix
    """Takes in a parameter folder, reads all csv files and then returns a list containing the csv files.
    """

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
