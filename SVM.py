__author__ = 'ericRincon'

from os import listdir
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np

""" takes in a list of csv files and then reads the specified column of each csv file defined by i. Will return a giant
    list that contains all the columns
"""
def read_column_as_list(csv_files, i):
    columns = []

    for csv_file in csv_files:
        column = csv_file.ix[:,i]
        col_as_list = column.tolist()
        columns = columns + col_as_list
    return columns

"""Takes in a parameter folder, reads all csv files and then returns a list containing the csv files.
"""
def read_csv_files(folder):
    csv_file_paths = listdir(folder)
    csv_file_data = []

    for csv_file_path in csv_file_paths:
        if '.csv' in csv_file_path:
            csv_data = read_csv(folder + csv_file_path)
            csv_file_data.append(csv_data)

    return csv_file_data

def main():
    x = listdir('datasets')
    csv_files = read_csv_files('datasets/')

    indices = [1, 4, 5] #Represents the ith column that must be read. 1: Title, 2: Abstract, and 4: M

    feature_matrix = [] #Contins all the columns title, abstract, and MESH

    for i in indices:
        column = read_column_as_list(csv_files, i)
        feature_matrix.append(column)

    count_vect = CountVectorizer()
    X_train_counts = []

    for vector in feature_matrix:
        temp_vector = count_vect.fit_transform(vector)
        print(temp_vector.shape)
        X_train_counts.append(temp_vector)


    X_train_counts_tf = []
    tf_transformer = TfidfTransformer(use_idf=False)
    for count_vector in X_train_counts:
        temp_count_vector = tf_transformer.fit_transform(count_vector)
        X_train_counts_tf.append(temp_count_vector)

    Y = read_column_as_list(csv_files, 6)
    Y = np.array(Y)
    print(Y)
    clf = SGDClassifier()
    clf.fit(X_train_counts_tf,Y)


if __name__ == '__main__':
    main()
