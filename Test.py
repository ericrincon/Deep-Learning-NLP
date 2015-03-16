__author__ = 'eric_rincon'

import numpy
import sys
import theano
import scipy.sparse

from sklearn.linear_model import Perceptron

from os import listdir
from os.path import isfile, join
from random import shuffle
from theano import tensor as T
from DataLoader import DataLoader
from Ensemble import Ensemble
from NeuralNet import NeuralNet
from Fold import Fold
from random import shuffle
from SVM import SVM

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from theano import tensor

def main():
    transfer_learning()
def test_svm_vs_mlp():
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
    #  values
    mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 75, 'batch_size': 20,
                       'activation_function': T.tanh, 'train_p':.6, 'n_layers': [100, 100]}
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
    #  values

    #create list of mlps
    path = 'datasets/'


    files = [f for f in listdir(path) if isfile(join(path,f))]
    files.pop(0) #remove the first element of files because of OSX file .DS_Store that exists in every directory
    indices = list(range(len(files)))
    shuffle(indices)

    for n in range(1):

        file_path = files[indices[n]]
        svm_output_file_name = "output/Multiple Hidden Layers/SVM" + file_path.split(".")[0] + ".txt"
        mlp_output_file_name = "output/Multiple Hidden Layers/MLP" + file_path.split(".")[0] + ".txt"
        svm_output_file = open(svm_output_file_name, 'w+')
        mlp_output_file = open(mlp_output_file_name, "w+")
        names = {1: 'title' ,4: 'abstract', 5:'mesh', 'y': 6}
        path+=file_path
        data_loader = DataLoader(path)
        data = data_loader.get_training_test_m(path, names, add_index_vector=False)
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
            mlp_fold.start()
            #Make sure that the both folds finish before moving on
            #Start threads for mlp and svm fold
            svm_fold.run()
            print(svm_fold)
            mlp_fold.join()
            print(mlp_fold)


        path = 'datasets/'


def test_mlp():
    path = 'datasets/'


    path = 'datasets/ACEInhibitors_processed.csv'
    data_loader = DataLoader(path)
    names = {1: 'title' ,4: 'abstract', 5:'mesh', 'y': 6}

    data = data_loader.get_training_test_m(path, names, add_index_vector=False)
    mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 75, 'batch_size': 20,
                       'activation_function': T.tanh, 'train_p':.6, 'n_layers': [2000, 1000]}
    data = data_loader.underSample(data[0], data[1])
    data = data_loader.create_random_samples(data[0], data[1], train_p=.8, test_p=.2)
    mlp = NeuralNet(parameter_list=mlp_parameters)
    mlp.train(data[0][0][:, 1:], data[1][0])
    mlp.test(data[0][1][:, 1:], data[1][1])
    print(mlp)

def transfer_learning():
    path = 'datasets/'
    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}
    svm_wda_metrics_list = []
    svm_metrics_list = []
    perceptron_metrics_list = []
    perceptron_modified_metrics_list = []

    #get all the data paths and shuffle them randomly
    files = [f for f in listdir(path) if isfile(join(path,f))]
    #remove the first element of files because of OSX file .DS_Store that exists in every directory
    files.pop(0)

    #Shuffle the files
    shuffle(files)

    data_loader = DataLoader(path)

    #Get one file out of the csv files in the dataloader use this as the held out domain
    held_out_domain = data_loader.csv_files.pop()

    #Get the feature representation of the held out data
    held_out_x, held_out_y = data_loader.get_feature_matrix(names, held_out_domain)
    #Create the folds for the held out data in this case the default 5
    folds = data_loader.cross_fold_valdation(held_out_x, held_out_y)

    mlp_parameters = {"learning_rate": 1.2, 'L1_term': 0.000, 'L2_term': .0001, 'n_epochs': 100, 'batch_size': 20,
                           'activation_function': T.tanh, 'train_p':.6, 'n_layers': [1500]}

    #Start the 5 fold cross validation
    for n_fold, fold in enumerate(folds):
        print("Fold {}: ".format(n_fold))
        #Each sample is a list that contains the x and y for the classifier
        #Typically fold[0] would be the train sample but because it is switched for
        #testing the effectiveness of the domain adaptation
        train_sample = fold[1]
        test_sample = fold[0]

        #These are the original copies to be copied over the augmented feature matrix
        #Each sample contains the text and y labels from the data before it is put into the sklearn count vectorizer
        target_source_pre_train_x = train_sample[0]
        target_source_train_y = train_sample[1]
        target_source_pre_test_x = test_sample[0]
        target_source_test_y = test_sample[1]

        #Get the bag of words representation of the small 20% target source data and transform the other 80%
        #of the data.
        target_source_train_x_copy = data_loader.get_transformed_features(target_source_pre_train_x, True, False)
        target_source_test_x_copy = data_loader.transform(target_source_pre_test_x, True)

        #Read the n-1 csv files
        source_x, source_y = data_loader.get_feature_matrix(names)
        #Get the total number of domains i.e., the number of files with documents
        n_source_domains = len(files)

        source_x = data_loader.transform(source_x, True)
        previous_domain_sizes = 0

        #Used to keep track of where to put the feature vector x in the augmented feature matrix
        current_domain_number = 0

        domain_size = data_loader.csv_files.pop().shape[0]
        i_start = previous_domain_sizes
        i_stop = domain_size
        previous_domain_sizes = domain_size
        source_domain_x = source_x[i_start:i_stop, :]
        source_domain_y = source_y[i_start:i_stop]
        source_domain_x, source_domain_y = data_loader.underSample(source_domain_x, source_domain_y)
        current_domain_number = 2
        current_domain_zero_matrix = numpy.zeros((source_domain_x.shape[0], source_domain_x.shape[1]))

        #Create the first part of the augmented feature matrix
        for i in range(n_source_domains - 1):
            source_domain_x = scipy.sparse.hstack((source_domain_x, current_domain_zero_matrix))

        #Create the rest of the augmented feature matrix and stack on to the first part
        for csv_file in data_loader.csv_files:
            domain_size = csv_file.shape[0]
            i_start = previous_domain_sizes

            i_stop = domain_size + previous_domain_sizes
            previous_domain_sizes += domain_size
            current_domain_x = source_x[i_start + 1:i_stop,:]
            current_domain_y = source_y[i_start + 1:i_stop]

            current_undersampled_domain_x, current_undersampled_domain_y = data_loader.underSample(
                current_domain_x, current_domain_y)

            current_domain_r = current_undersampled_domain_x.shape[0]
            current_domain_c = current_undersampled_domain_x.shape[1]

            current_domain_zero_matrix = numpy.zeros((current_domain_r, current_domain_c))

            current_domain_matrix = current_undersampled_domain_x

            for i in range(len(files) - 1 ):

                if i == current_domain_number:
                    current_domain_matrix = scipy.sparse.hstack((current_domain_matrix,
                        current_undersampled_domain_x))
                else:
                    current_domain_matrix = scipy.sparse.hstack((current_domain_matrix,
                        current_domain_zero_matrix))
            current_domain_number += 1
            source_domain_x = scipy.sparse.vstack((source_domain_x, current_domain_matrix))
            source_domain_y = numpy.hstack((source_domain_y, current_undersampled_domain_y))


        target_source_train_x = target_source_train_x_copy
        target_source_test_x = target_source_test_x_copy
        #Create the augmented matrix for the target data i.e. phi(x) = <x,0.....,0,x>
        target_source_zero_matrix_train = numpy.zeros((target_source_train_x.shape[0], target_source_train_x.shape[1]))
        target_source_zero_matrix_test = numpy.zeros((target_source_test_x.shape[0], target_source_test_x.shape[1]))

        for i in range(len(files) - 1):
            if not i == (len(files) - 1):
                target_source_train_x = scipy.sparse.hstack((target_source_train_x, target_source_zero_matrix_train))
                target_source_test_x = scipy.sparse.hstack((target_source_test_x, target_source_zero_matrix_test))
            else:
                target_source_train_x = scipy.sparse.hstack((target_source_train_x, target_source_train_x_copy))
                target_source_test_x = scipy.sparse.hstack((target_source_test_x, target_source_test_x_copy))


        #Shave some random examples because of SGD for the MLP and the batch size of 10, 20, etc.
        index_stop = source_domain_x.shape[0] - (source_domain_x.shape[0] % 10)

        mlp_train_x = source_domain_x.tocsr()[:index_stop, :]

        mlp_train_x = mlp_train_x.tocsc()[:, :10000].todense()
        mlp_train_y = source_domain_y[:index_stop]

        train_x = scipy.sparse.vstack((source_domain_x, target_source_train_x))
        train_y = numpy.hstack((source_domain_y, target_source_train_y))

        test_x = target_source_test_x
        test_y = target_source_test_y

        perceptron_train_x = target_source_train_x_copy.todense()
        perceptron_train_y = target_source_train_y

        perceptron_test_x = target_source_test_x.todense()
        perceptron_test_y = target_source_test_y


        #SVM with the augmented feature matrix for domain adaptation
        svm_wda = SVM()
        svm_wda.train(train_x, train_y)
        svm_wda.test(test_x, test_y)
        print("SVM with domain adaptation metrics:")
        print(svm_wda)
        print("\n")
        svm_wda_metrics_list.append(svm_wda.metrics)

        #SVM without the domain adaptation
        svm = SVM()
        svm.train(target_source_train_x_copy, target_source_train_y)
        svm.test(target_source_test_x_copy, target_source_test_y)
        print("SVM without domain adaptation")
        print(svm)
        print("\n")
        svm_metrics_list.append(svm.metrics)

        classifier = NeuralNet(parameter_list=mlp_parameters)
        classifier.train(mlp_train_x, mlp_train_y)
        #get the weights and bias values of the trained MLPs hidden layer
        W = classifier.mlp.hidden_layers[0].W
        b = classifier.mlp.hidden_layers[0].b

        #Transform the feature vectors of the held out data to the learned hidden layer features of the previous
        #MLP trained with all n-1 datasets

        a_function = mlp_parameters['activation_function']
        perceptron_train_x = theano.shared(perceptron_train_x)
        perceptron_test_x = theano.shared(perceptron_test_x)
        transformation_function = theano.function(
            inputs=[],
            outputs=[a_function(T.dot(perceptron_train_x, W) + b),
                     a_function(T.dot(perceptron_test_x, W) + b)],
            on_unused_input='ignore',

        )

        transformed_perceptron_train_x, transformed_perceptron_test_x = transformation_function()
        modified_transformed_perceptron_train_x = numpy.hstack((transformed_perceptron_train_x, target_source_train_x_copy.todense()))
        modified_transformed_perceptron_test_x = numpy.hstack((transformed_perceptron_test_x, target_source_test_x_copy.todense()))

        perceptron = Perceptron(penalty="l2", n_iter=100)
        perceptron.fit(transformed_perceptron_train_x, perceptron_train_y)
        prediction = perceptron.predict(transformed_perceptron_test_x)


        perceptron_metrics = {
            "f1": f1_score(perceptron_test_y, prediction),
            "precision": precision_score(perceptron_test_y, prediction),
            "recall": recall_score(perceptron_test_y, prediction),
            "roc": roc_auc_score(perceptron_test_y, prediction),
            "accuracy": accuracy_score(perceptron_test_y, prediction)
        }

        perceptron_metrics_list.append(perceptron_metrics)

        print("Perceptron with the transformed features")
        print_classifier_scores(perceptron_metrics)
        print("\n")
        perceptron_modified = Perceptron(penalty="l2", n_iter=100)
        perceptron_modified.fit(modified_transformed_perceptron_train_x, perceptron_train_y)
        prediction_modified = perceptron_modified.predict(modified_transformed_perceptron_test_x)

        perceptron_metrics_modified = {
            "f1": f1_score(perceptron_test_y, prediction_modified),
            "precision": precision_score(perceptron_test_y, prediction_modified),
            "recall": recall_score(perceptron_test_y, prediction_modified),
            "roc": roc_auc_score(perceptron_test_y, prediction_modified),
            "accuracy": accuracy_score(perceptron_test_y, prediction_modified)
        }
        perceptron_modified_metrics_list.append(perceptron_metrics_modified)

        print("Perceptron with the transformed features and concatenated bag of words ")
        print_classifier_scores(perceptron_metrics_modified)

        print("*********** End of fold {} ***********".format(n_fold))
    print("Fold Scores")
    print("SVM with domain adaptation")
    calculate_and_print_fold_scores(svm_wda_metrics_list)


def calculate_and_print_fold_scores(metrics_list):
    keys = metrics_list[0].keys()

    perceptron_metrics_modified = {
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "roc": 0,
            "accuracy": 0
    }

    for metrics_dict in metrics_list:
        for key in keys:
            perceptron_metrics_modified[key] = perceptron_metrics_modified[key] + metrics_dict[key]
    for key in perceptron_metrics_modified:
        perceptron_metrics_modified[key] = perceptron_metrics_modified[key]/len(metrics_list)

    print_classifier_scores(perceptron_metrics_modified)


    print()
def print_classifier_scores(metrics):
    keys = metrics.keys()
    for key in keys:
        output = "{}: {}".format(key, metrics[key])
        print(output)

if __name__ == "__main__":
    main()
