__author__ = 'eric_rincon'

import numpy
import sys

from os import listdir
from os.path import isfile, join
from random import shuffle
from theano import tensor as T
from DataLoader import DataLoader
from Ensemble import Ensemble
from NeuralNet import NeuralNet
from Fold import Fold

def main():
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
    #  values
    mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 75, 'batch_size': 20,
                      'n_hidden_units': 1000, 'activation_function': T.tanh, 'n_layers': 1, 'train_p':.6}
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
    #  values

    #create list of mlps

    path = 'datasets/'


    files = [f for f in listdir(path) if isfile(join(path,f))]
    files.pop(0) #remove the first element of files because of OSX file .DS_Store that exists in every directory
    indices = list(range(len(files)))
    shuffle(indices)

    for n in range(3):
        file_path = files[indices[n]]
        print(file_path)
        svm_output_file_name = "output/SVM" + file_path.split(".")[0] + ".txt"
        mlp_output_file_name = "output/MLP" + file_path.split(".")[0] + ".txt"
        svm_output_file = open(svm_output_file_name, 'w+')
        mlp_output_file = open(mlp_output_file_name, "w+")
        names = {1: 'title' ,4: 'abstract', 5:'mesh', 'y': 6}
        path+=file_path
        data_loader = DataLoader(path)
        data = data_loader.get_training_test_m(path, names)
        folds = data_loader.cross_fold_valdation(data[0], data[1])

        k_folds = numpy.arange(len(folds))
        #Ensemble defaults to classifier type SVM with kernel RBF and will gridsearch
        svm_ensemble = Ensemble()

        mlps = []

        for n in range(11):
            mlps.append(NeuralNet(parameter_list=mlp_parameters))
        mlp_ensemble = Ensemble(list_of_classifiers=mlps)

        for k in k_folds:
            train_sample = folds[k][0]
            test_sample = folds[k][1]
            #train_sample contains x: features, and y: labels to train the classifier
            #test_sample also contains an x, and y but

            svm_fold = Fold(n, train_sample, test_sample, svm_ensemble, svm_output_file)
            mlp_fold = Fold(n, train_sample, test_sample, mlp_ensemble, mlp_output_file)

            #Start threads for mlp and svm fold
            svm_fold.start()
            mlp_fold.start()

            #Make sure that the both folds finish before moving on
            svm_fold.join()
            mlp_fold.join()
            print(svm_fold)
            print(mlp_fold)


if __name__ == "__main__":
    main()
