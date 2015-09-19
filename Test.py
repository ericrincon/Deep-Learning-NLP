__author__ = 'eric_rincon'

import numpy
import theano
import copy
import os
import sys
from Feature_Engineer import Feature_Engineer
from sklearn.linear_model import Perceptron

from os import listdir
from os.path import isfile, join
from DataLoader import DataLoader
from NeuralNet import NeuralNet
from SVM import SVM

from scipy import sparse
from csv import DictWriter
import csv

def main():
    transfer_learning()

def transfer_learning(print_output=True):
    path = 'datasets/'
    data_loader = DataLoader(path)
    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}
    transformed_data_sets = []

    path = 'datasets/'

    files = [f for f in listdir(path) if isfile(join(path,f))]
    files.pop(0)
    data_loader = DataLoader(path)
    domains = data_loader.csv_files
    all_domains = copy.deepcopy(domains)
    training_domains = data_loader.csv_files
    all_domains_svm_wda_metrics_list = []
    all_domains_svm_metrics_list = []
    all_domains_svm_bow_mlp_list = []
    all_domains_mlp_fold_scores = []

    for i, held_out_domain in enumerate(domains):
        training_domains.pop(i)
        names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}
        svm_wda_metrics_list = []
        svm_metrics_list = []
        svm_bow_mlp_list = []

        folder_name = '/' + files[i]
        domain_name = files[i].__str__()
        domain_name = domain_name.split('.')[0]
        folder_name = 'output' + '/' + domain_name

        output = "Dataset: {}".format(files[i])
        if print_output:
            print(output)

        #shuffle(data_loader.csv_files)
        data_loader.csv_files = training_domains
        data_sets = data_loader.csv_files
        domains = data_loader.get_feature_matrix(names)

        #Get one file out of the csv files in the dataloader use this as the held out domain

        #Get the feature representation of the held out data
        held_out_x, held_out_y = data_loader.get_feature_matrix(names, held_out_domain)
        #Create the folds for the held out data in this case the default 5
        folds = data_loader.cross_fold_valdation(held_out_x, held_out_y)
        #Get the total number of domains i.e., the number of files with documents
        n_source_domains = len(data_sets)
        os.makedirs(folder_name)

        #Must convert the data type of the matrix for theano
        feature_engineer = Feature_Engineer()

        #Start the 5 fold cross validation
        for n_fold, fold in enumerate(folds):
            output = "Fold {}: \n".format(n_fold)
            if print_output:
                print(output)
            output = '{}/{}/fold_{}.csv'.format(os.getcwd(), folder_name, (n_fold + 1))
            file = open(output, 'w')
            csv_writer = csv.writer(file)

            #Each sample is a list that contains the x and y for the classifier
            #Typically fold[0] would be the train sample but because it is switched for
            #testing the effectiveness of the domain adaptation
            train_sample = fold[1]
            test_sample = fold[0]

            #These are the original copies to be copied over the augmented feature matrix
            #Each sample contains the text and y labels from the data before it is put into the sklearn count vectorizer
            train_x, train_y = train_sample
            test_x, test_y = test_sample

            train_y[train_y == 0] = 2
            train_y[train_y == 1] = 3
            test_y[test_y == 0] = 2
            test_y[test_y == 1] = 3


            #Get the bag of words representation of the small 20% target source data and transform the other 80%
            #of the data.
            train_x = data_loader.get_transformed_features(train_x, True, False, True)
            test_x = data_loader.transform(test_x, True, True)

            transformed_domains = []

            #Transform the domains with respect to the training data
            for domain in domains:
                domain_x, domain_y = domain
                transformed_domain_x = data_loader.transform(domain_x, True, True)
                transformed_domain_x, domain_y = data_loader.underSample(transformed_domain_x, domain_y)
                transformed_domains.append([transformed_domain_x, domain_y])

            augmented_feature_matrix_train, augmented_y_train = feature_engineer.augmented_feature_matrix(transformed_domains,
                                                                                              [train_x, train_y])
            augmented_feature_matrix_test, augmented_y_test = feature_engineer.augmented_feature_matrix(held_out_domain=[test_x, test_y],
                                                                                                        train_or_test=False,
                                                                                                        n_source_domains=len(transformed_domains))
            augmented_y_test[augmented_y_test == 2] = 0
            augmented_y_test[augmented_y_test == 3] = 1
            #SVM with the augmented feature matrix for domain adaptation
            svm_wda = SVM()
            svm_wda.train(augmented_feature_matrix_train, augmented_y_train)
            svm_wda.test(augmented_feature_matrix_test, augmented_y_test)
            output = "\nSVM with domain adaptation metrics:"
            csv_writer.writerow([output])
            if print_output:
                print(output)
                print(svm_wda)
                print("\n")
            svm_wda_metrics_list.append(svm_wda.metrics)

            classifier = NeuralNet(n_hidden_units=[250], output_size=4, batch_size=20, n_epochs=200, dropout=True,
                                   activation_function='relu', learning_rate=.3, momentum=True, momentum_term=.5)
            write_to_csv(svm_wda.metrics, csv_writer)


            y_for_mlp = []
            #Set up the x and y data for the MLP
            for p, domain in enumerate(transformed_domains):
                domain_x, domain_y = domain
                domain_x = domain_x.todense()
                y_for_mlp.append(domain_y)

                if p == 0:
                    neural_net_x_train = domain_x
                    neural_net_y_train = domain_y
                else:
                    neural_net_x_train = numpy.vstack((neural_net_x_train, domain_x))
                    neural_net_y_train = numpy.hstack((neural_net_y_train, domain_y))

            neural_net_x_train = numpy.float_(neural_net_x_train)


            classifier.train(neural_net_x_train, neural_net_y_train)

            test_y[test_y == 2] = 0
            test_y[test_y == 3] = 1
            svm_y_train = neural_net_y_train
            svm_y_train[svm_y_train == 2] = 0
            svm_y_train[svm_y_train == 3] = 1

            #SVM without the domain adaptation
            svm = SVM()
            svm.train(sparse.coo_matrix(neural_net_x_train), svm_y_train)
            svm.test(test_x, test_y)
            output = "\nSVM without domain adaptation"
            if print_output:
                print(output)
                print(svm)
                print("\n")
            csv_writer.writerow([output])
            svm_metrics_list.append(svm.metrics)
            write_to_csv(svm.metrics, csv_writer)


            #Transform the feature vectors of the held out data to the learned hidden layer features of the previous
            #MLP trained with all n-1 datasets

            perceptron_train_x = theano.shared(neural_net_x_train)
            perceptron_test_x = theano.shared(test_x.todense())

            transformed_perceptron_train_x = classifier.transfer_learned_weights(perceptron_train_x)
            transformed_perceptron_test_x = classifier.transfer_learned_weights(perceptron_test_x)

            modified_transformed_perceptron_train_x = numpy.hstack((transformed_perceptron_train_x,
                                                                    neural_net_x_train))
            modified_transformed_perceptron_test_x = numpy.hstack((transformed_perceptron_test_x,
                                                                   test_x.todense()))

            output = "\nSVM with BoW and transformed features"
            csv_writer.writerow([output])
            if print_output:
                print(output)
            svm_mlp_bow = SVM()
            svm_mlp_bow.train(sparse.coo_matrix(modified_transformed_perceptron_train_x), svm_y_train)
            svm_mlp_bow.test(sparse.coo_matrix(modified_transformed_perceptron_test_x), test_y)
            write_to_csv(svm_mlp_bow.metrics, csv_writer)
            if print_output:
                print(svm_mlp_bow)
            svm_bow_mlp_list.append(svm_mlp_bow.metrics)


            output = "*********** End of fold {} ***********".format(n_fold)

            if print_output:
                print(output)


        training_domains = copy.deepcopy(all_domains)
        file_name = '{}/{}/fold_averages.csv'.format(os.getcwd(), folder_name)
        file = open(file_name, 'w+')
        csv_writer = csv.writer(file)

        if print_output:
            output = "----------------------------------------------------------------------------------------" \
                     "\nFold Scores\n " \
                     "SVM with domain adaptation"
            print_write_output(output, svm_wda_metrics_list, all_domains_svm_wda_metrics_list, csv_writer)

            output = "\nSVM without domain adaptation"
            print_write_output(output, svm_metrics_list, all_domains_svm_metrics_list, csv_writer)

            output = "SVM with BoW and transformed features"
            print_write_output(output, svm_bow_mlp_list, all_domains_svm_bow_mlp_list, csv_writer)



    file_name = '{}/output/all_fold_averages.csv'.format(os.getcwd())
    file = open(file_name, 'w+')
    csv_writer = csv.writer(file)
    if print_output:
        output = "*******************************************************************************************" \
                 "\nAll domain macro metric scores\n " \
                 "SVM with domain adaptation"
        print_macro_scores("SVM with domain adaptation", all_domains_svm_wda_metrics_list, csv_writer)

        output = "\nSVM without domain adaptation"
        print_macro_scores(output, all_domains_svm_metrics_list, csv_writer)

        output = "SVM with BoW and transformed features"
        print_macro_scores(output, all_domains_svm_bow_mlp_list, csv_writer)





def print_write_output(title, classifier_metrics, all_domain_metrics, csv_writer):
    csv_writer.writerow([title])
    print(title)
    metrics = calculate_fold_scores(classifier_metrics)
    all_domain_metrics.append(metrics)
    output_classifer = print_classifier_scores(metrics)
    print(output_classifer)
    write_to_csv(metrics, csv_writer)
def print_macro_scores(output, classifier_metrics, csv_writer):
    csv_writer.writerow([output])
    print(output)
    metrics = calculate_fold_scores(classifier_metrics)
    output_classifer = print_classifier_scores(metrics)
    print(output_classifer)
    write_to_csv(metrics, csv_writer)

def calculate_fold_scores(metrics_list):
    keys = metrics_list[0].keys()

    temp_metrics_dict = {}
    for key in keys:
        temp_metrics_dict.update({key: 0})
    for metrics_dict in metrics_list:
        for key in keys:
            temp_metrics_dict[key] = temp_metrics_dict[key] + metrics_dict[key]
    for key in temp_metrics_dict:
        temp_metrics_dict[key] = temp_metrics_dict[key]/len(metrics_list)

    return temp_metrics_dict

def print_classifier_scores(metrics):
    string = ''

    keys = metrics.keys()
    for key in keys:
        output = "{}: {}\n".format(key, metrics[key])
        string+=output
    return string


def write_to_csv(dict, csv_writer):
    for key, value in dict.items():
        csv_writer.writerow([key, value])

if __name__ == "__main__":
    main()
