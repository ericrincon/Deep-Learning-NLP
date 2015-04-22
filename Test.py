__author__ = 'eric_rincon'

import numpy
import theano
import scipy.sparse
import copy

from Feature_Engineer import Feature_Engineer
from sklearn.linear_model import Perceptron

from os import listdir
from os.path import isfile, join
from theano import tensor as T
from DataLoader import DataLoader
from Ensemble import Ensemble
from NeuralNet import NeuralNet
from Fold import Fold
from random import shuffle
from SVM import SVM
from Domain_MLP import Domain_MLP

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from scipy import sparse

def main():
    #transfer_learning()
    transfer_learning_new()

def transfer_learning_new():
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
    file = open('test.txt', 'w+')
    for i, held_out_domain in enumerate(domains):
        training_domains.pop(i)
        names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}
        svm_wda_metrics_list = []
        svm_metrics_list = []
        perceptron_metrics_list = []
        perceptron_modified_metrics_list = []
        perceptron_without_list = []
        svm_bow_mlp_list = []
        mlp_fold_scores = []
        print("Dataset: {}".format(files[i]))
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


        #Must convert the data type of the matrix for theano
        feature_engineer = Feature_Engineer()

        #Start the 5 fold cross validation
        for n_fold, fold in enumerate(folds):
            output = "Fold {}: \n".format(n_fold)
            print(output)
            file.write(output)

            #Each sample is a list that contains the x and y for the classifier
            #Typically fold[0] would be the train sample but because it is switched for
            #testing the effectiveness of the domain adaptation
            train_sample = fold[1]
            test_sample = fold[0]

            #These are the original copies to be copied over the augmented feature matrix
            #Each sample contains the text and y labels from the data before it is put into the sklearn count vectorizer
            train_x, train_y = train_sample
            test_x, test_y = test_sample

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

            #SVM with the augmented feature matrix for domain adaptation
            svm_wda = SVM()
            svm_wda.train(augmented_feature_matrix_train, augmented_y_train)
            svm_wda.test(augmented_feature_matrix_test, augmented_y_test)
            output = "SVM with domain adaptation metrics:"
            print(output)
            print(svm_wda)
            print("\n")
            file.write(output)
            file.write(svm_wda.__str__())
            file.write('\n')
            svm_wda_metrics_list.append(svm_wda.metrics)
            classifier = NeuralNet(n_hidden_units=[250, 100], output_size=2, batch_size=20, n_epochs=200, dropout=True,
                                   activation_function='relu', learning_rate=.3, momentum=True, momentum_term=.5)
            classifier_tanh = NeuralNet(n_hidden_units=[250, 100], output_size=2, batch_size=20, n_epochs=200, dropout=True,
                                   activation_function='tanh', learning_rate=.3, momentum=True, momentum_term=.5)



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
            #SVM without the domain adaptation
            svm = SVM()
            svm.train(sparse.coo_matrix(neural_net_x_train), neural_net_y_train)
            svm.test(test_x, test_y)
            output = "SVM without domain adaptation"
            print(output)
            print(svm)
            print("\n")
            svm_metrics_list.append(svm.metrics)
            file.write(output)
            file.write(svm.__str__())
            file.write('\n')
            output = "Perceptron with only bag of words"
            print(output)
            file.write(output)

            perceptron_without = Perceptron(penalty='l2', n_iter=100)
            perceptron_without.fit(neural_net_x_train, neural_net_y_train)
            perceptron_without_prediction = perceptron_without.predict(test_x)
            perceptron_metrics_modified_bow = {
                "f1": f1_score(test_y, perceptron_without_prediction),
                "precision": precision_score(test_y, perceptron_without_prediction),
                "recall": recall_score(test_y, perceptron_without_prediction),
                "roc": roc_auc_score(test_y, perceptron_without_prediction),
                "accuracy": accuracy_score(test_y, perceptron_without_prediction)
            }
            perceptron_without_list.append(perceptron_metrics_modified_bow)

            neural_net_x_train = numpy.float_(neural_net_x_train)
            classifier_tanh.train(neural_net_x_train, neural_net_y_train)
            classifier_tanh.test(test_x.todense(), test_y)
            print("Tanh MLP")
            print(classifier_tanh)
            classifier.train(neural_net_x_train, neural_net_y_train)
            classifier.test(test_x.todense(), test_y)
            print(classifier)
            mlp_fold_scores.append(classifier.metrics)
            file.write(classifier.__str__())
            #get the weights and bias values of the trained MLPs hidden layer
            W = classifier.mlp.hidden_layers[0].W
            b = classifier.mlp.hidden_layers[0].b



            #Transform the feature vectors of the held out data to the learned hidden layer features of the previous
            #MLP trained with all n-1 datasets

            a_function = classifier.activation_function
            perceptron_train_x = theano.shared(neural_net_x_train)
            perceptron_test_x = theano.shared(test_x.todense())
            if classifier.dropout:
                transformation_function = theano.function(
                    inputs=[],
                    outputs=[a_function(T.dot(perceptron_train_x, (W * classifier.dropout_rate)) + b),
                             a_function(T.dot(perceptron_test_x, (W * classifier.dropout_rate)) + b)],
                    on_unused_input='ignore',

                )
            else:
                transformation_function = theano.function(
                    inputs=[],
                    outputs=[a_function(T.dot(perceptron_train_x, W) + b),
                             a_function(T.dot(perceptron_test_x, W) + b)],
                    on_unused_input='ignore',

                )

            transformed_perceptron_train_x, transformed_perceptron_test_x = transformation_function()
            modified_transformed_perceptron_train_x = numpy.hstack((transformed_perceptron_train_x,
                                                                    neural_net_x_train))
            modified_transformed_perceptron_test_x = numpy.hstack((transformed_perceptron_test_x,
                                                                   test_x.todense()))

            perceptron = Perceptron(penalty="l2", n_iter=100)
            perceptron.fit(transformed_perceptron_train_x, neural_net_y_train)
            prediction = perceptron.predict(transformed_perceptron_test_x)
            output = "\nSVM with BoW and transformed features"
            print(output)
            svm_mlp_bow = SVM()
            svm_mlp_bow.train(modified_transformed_perceptron_train_x, neural_net_y_train)
            svm_mlp_bow.test(modified_transformed_perceptron_test_x, test_y)
            print(svm_mlp_bow)
            file.write(svm_mlp_bow.__str__())
            svm_bow_mlp_list.append(svm_mlp_bow.metrics)

            perceptron_metrics = {
                "f1": f1_score(test_y, prediction),
                "precision": precision_score(test_y, prediction),
                "recall": recall_score(test_y, prediction),
                "roc": roc_auc_score(test_y, prediction),
                "accuracy": accuracy_score(test_y, prediction)
            }

            perceptron_metrics_list.append(perceptron_metrics)
            output = "Perceptron with the transformed features"
            print(output)
            file.write(output)
            file.write(print_classifier_scores(perceptron_metrics))
            print(print_classifier_scores(perceptron_metrics))
            print("\n")
            file.write('\n')
            perceptron_modified = Perceptron(penalty="l2", n_iter=100)
            perceptron_modified.fit(modified_transformed_perceptron_train_x, neural_net_y_train)
            prediction_modified = perceptron_modified.predict(modified_transformed_perceptron_test_x)

            perceptron_metrics_modified = {
                "f1": f1_score(test_y, prediction_modified),
                "precision": precision_score(test_y, prediction_modified),
                "recall": recall_score(test_y, prediction_modified),
                "roc": roc_auc_score(test_y, prediction_modified),
                "accuracy": accuracy_score(test_y, prediction_modified)
            }
            perceptron_modified_metrics_list.append(perceptron_metrics_modified)
            output = "Perceptron with the transformed features and concatenated bag of words "
            print(output)
            file.write(output)
            file.write(print_classifier_scores(perceptron_metrics_modified))
            print(print_classifier_scores(perceptron_metrics_modified))

            print("*********** End of fold {} ***********".format(n_fold))

        training_domains = copy.deepcopy(all_domains)

        print("Fold Scores")
        print("SVM with domain adaptation")

        print(calculate_and_print_fold_scores(svm_wda_metrics_list))
        print("\nSVM without domain adaptation")
        print(calculate_and_print_fold_scores(svm_metrics_list))
        print("\nPerceptron without transfer learning")
        print(calculate_and_print_fold_scores(perceptron_without_list))
        print("\nPerceptron with transfer learning")
        print(calculate_and_print_fold_scores(perceptron_metrics_list))
        print("\nPerceptron with transfer learning and concatenated bag of words")
        print(calculate_and_print_fold_scores(perceptron_modified_metrics_list))
        print("SVM with BoW and transformed features")
        print(calculate_and_print_fold_scores(svm_bow_mlp_list))
        print("MLP scores")
        print(calculate_and_print_fold_scores(mlp_fold_scores))
        file.write(calculate_and_print_fold_scores(svm_wda_metrics_list))
        file.write("\nSVM without domain adaptation")
        file.write(calculate_and_print_fold_scores(svm_metrics_list))
        file.write("\nPerceptron without transfer learning")
        file.write(calculate_and_print_fold_scores(perceptron_without_list))
        file.write("\nPerceptron with transfer learning")
        file.write(calculate_and_print_fold_scores(perceptron_metrics_list))
        file.write("\nPerceptron with transfer learning and concatenated bag of words")
        file.write(calculate_and_print_fold_scores(perceptron_modified_metrics_list))
        file.write("SVM with BoW and transformed features")
        file.write(calculate_and_print_fold_scores(svm_bow_mlp_list))
        file.write("MLP scores")
        file.write(calculate_and_print_fold_scores(mlp_fold_scores))




def test_domain_mlp():
    mlp = Domain_MLP()

def test_svm_vs_mlp():
    #names: represents the ith column that must be read. 1: Title, 2: Abstract, 4: MESH, and y: is the column for the target
    #  values

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

    data = data_loader.underSample(data[0], data[1])
    data = data_loader.create_random_samples(data[0], data[1], train_p=.8, test_p=.2)
    """
    mlp = NeuralNet(parameter_list=mlp_parameters)
    mlp.train(data[0][0][:, 1:], data[1][0])
    mlp.test(data[0][1][:, 1:], data[1][1])
    print(mlp)
    """
def transfer_learning_old():
    path = 'datasets/'
    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}
    svm_wda_metrics_list = []
    svm_metrics_list = []
    perceptron_metrics_list = []
    perceptron_modified_metrics_list = []
    perceptron_without_list = []
    svm_bow_mlp_list = []

    x = numpy.asarray([[1, 2, 3],[4, 5, 6]])

    #get all the data paths and shuffle them randomly
    files = [f for f in listdir(path) if isfile(join(path,f))]
    #remove the first element of files because of OSX file .DS_Store that exists in every directory
    files.pop(0)

    held_out_domain_index = files.index("ADHD_processed.csv")
    #Shuffle the files
   # print(files[0])

    data_loader = DataLoader(path)
    held_out_domain = data_loader.csv_files.pop(held_out_domain_index)
    shuffle(data_loader.csv_files)


    #Get one file out of the csv files in the dataloader use this as the held out domain
    #held_out_domain = data_loader.csv_files.pop()

    #Get the feature representation of the held out data
    held_out_x, held_out_y = data_loader.get_feature_matrix(names, held_out_domain)
    #Create the folds for the held out data in this case the default 5
    folds = data_loader.cross_fold_valdation(held_out_x, held_out_y)
    #Get the total number of domains i.e., the number of files with documents
    n_source_domains = len(files)
    mlp_parameters = {"learning_rate": 1.2, 'L1_term': 0.000, 'L2_term': .0001, 'n_epochs': 500, 'batch_size': 20,
                           'activation_function': T.tanh, 'train_p':.6, 'n_layers': [1500], "output_size": n_source_domains*2}


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


        source_x = data_loader.transform(source_x, True)
        previous_domain_sizes = 0

        #Used to keep track of where to put the feature vector x in the augmented feature matrix
        current_domain_number = 0

        domain_size = data_loader.csv_files.pop().shape[0]
        i_start = previous_domain_sizes

        i_stop = domain_size
        previous_domain_sizes = i_stop
        source_domain_x = source_x[i_start:i_stop, :]
        source_domain_y = source_y[i_start:i_stop]
        source_domain_x, source_domain_y = data_loader.underSample(source_domain_x, source_domain_y)
        current_domain_number = 2
        total = source_domain_x.shape[0]

        current_domain_zero_matrix = numpy.zeros((source_domain_x.shape[0], source_domain_x.shape[1]))

        #Create the first part of the augmented feature matrix
        for i in range(n_source_domains - 1):
            source_domain_x = scipy.sparse.hstack((source_domain_x, current_domain_zero_matrix))

        last_zero_index = 2
        last_one_index = 3

        #Create the rest of the augmented feature matrix and stack on to the first part
        for csv_file in data_loader.csv_files:
            domain_size = csv_file.shape[0]
            i_start = previous_domain_sizes

            i_stop = domain_size + i_start
            previous_domain_sizes += domain_size
            current_domain_x = source_x[i_start: i_stop, :]
            current_domain_y = source_y[i_start: i_stop]
            current_undersampled_domain_x, current_undersampled_domain_y = data_loader.underSample(
                current_domain_x, current_domain_y)

            current_undersampled_domain_y[current_undersampled_domain_y == 0] = last_zero_index
            current_undersampled_domain_y[current_undersampled_domain_y == 1] = last_one_index

            last_zero_index += 2
            last_one_index += 2

            current_domain_r = current_undersampled_domain_x.shape[0]
            current_domain_c = current_undersampled_domain_x.shape[1]
            total = total + current_domain_r

            current_domain_zero_matrix = numpy.zeros((current_domain_r, current_domain_c))

            current_domain_matrix = current_undersampled_domain_x

            for i in range(len(files) - 1):

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

        perceptron_test_x = target_source_test_x_copy.todense()
        test_y = target_source_test_y


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

        print("Perceptron with only bag of words")
        perceptron_without = Perceptron(penalty='l2', n_iter=100)
        perceptron_without.fit(target_source_train_x_copy, target_source_train_y)
        perceptron_without_prediction = perceptron_without.predict(target_source_test_x_copy)
        perceptron_metrics_modified_bow = {
            "f1": f1_score(target_source_test_y, perceptron_without_prediction),
            "precision": precision_score(target_source_test_y, perceptron_without_prediction),
            "recall": recall_score(target_source_test_y, perceptron_without_prediction),
            "roc": roc_auc_score(target_source_test_y, perceptron_without_prediction),
            "accuracy": accuracy_score(target_source_test_y, perceptron_without_prediction)
        }
        perceptron_without_list.append(perceptron_metrics_modified_bow)

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
        modified_transformed_perceptron_train_x = numpy.hstack((transformed_perceptron_train_x,
                                                                target_source_train_x_copy.todense()))
        modified_transformed_perceptron_test_x = numpy.hstack((transformed_perceptron_test_x,
                                                               target_source_test_x_copy.todense()))

        perceptron = Perceptron(penalty="l2", n_iter=100)
        perceptron.fit(transformed_perceptron_train_x, perceptron_train_y)
        prediction = perceptron.predict(transformed_perceptron_test_x)
        print("\nSVM with BoW and transformed features")
        svm_mlp_bow = SVM()
        svm_mlp_bow.train(modified_transformed_perceptron_train_x, perceptron_train_y)
        svm_mlp_bow.test(modified_transformed_perceptron_test_x, test_y)
        print(svm_mlp_bow)
        svm_bow_mlp_list.append(svm_mlp_bow.metrics)

        perceptron_metrics = {
            "f1": f1_score(test_y, prediction),
            "precision": precision_score(test_y, prediction),
            "recall": recall_score(test_y, prediction),
            "roc": roc_auc_score(test_y, prediction),
            "accuracy": accuracy_score(test_y, prediction)
        }

        perceptron_metrics_list.append(perceptron_metrics)
        output = "Perceptron with the transformed features"
        print(output)
        print(print_classifier_scores(perceptron_metrics))
        file.write(output)
        file.write(print_classifier_scores(perceptron_metrics))
        file.write('\n')
        print("\n")
        perceptron_modified = Perceptron(penalty="l2", n_iter=100)
        perceptron_modified.fit(modified_transformed_perceptron_train_x, perceptron_train_y)
        prediction_modified = perceptron_modified.predict(modified_transformed_perceptron_test_x)

        perceptron_metrics_modified = {
            "f1": f1_score(test_y, prediction_modified),
            "precision": precision_score(test_y, prediction_modified),
            "recall": recall_score(test_y, prediction_modified),
            "roc": roc_auc_score(test_y, prediction_modified),
            "accuracy": accuracy_score(test_y, prediction_modified)
        }
        perceptron_modified_metrics_list.append(perceptron_metrics_modified)

        print("Perceptron with the transformed features and concatenated bag of words ")
        print_classifier_scores(perceptron_metrics_modified)

        print("*********** End of fold {} ***********".format(n_fold))
        print("Fold Scores")
        print("SVM with domain adaptation")
        calculate_and_print_fold_scores(svm_wda_metrics_list)
        print("\nSVM without domain adaptation")
        calculate_and_print_fold_scores(svm_metrics_list)
        print("\nPerceptron without transfer learning")
        calculate_and_print_fold_scores(perceptron_without_list)
        print("\nPerceptron with transfer learning")
        calculate_and_print_fold_scores(perceptron_metrics_list)
        print("\nPerceptron with transfer learning and concatenated bag of words")
        calculate_and_print_fold_scores(perceptron_modified_metrics_list)
        print("SVM with BoW and transformed features")
        calculate_and_print_fold_scores(svm_bow_mlp_list)
        print(calculate_and_print_fold_scores(mlp_fold_scores))
        file.write(calculate_and_print_fold_scores(mlp_fold_scores))




def calculate_and_print_fold_scores(metrics_list):
    keys = metrics_list[0].keys()

    temp_metrics_dict = {}
    for key in keys:
        temp_metrics_dict.update({key: 0})
    for metrics_dict in metrics_list:
        for key in keys:
            temp_metrics_dict[key] = temp_metrics_dict[key] + metrics_dict[key]
    for key in temp_metrics_dict:
        temp_metrics_dict[key] = temp_metrics_dict[key]/len(metrics_list)

    return print_classifier_scores(temp_metrics_dict)

def print_classifier_scores(metrics):
    string = ''

    keys = metrics.keys()
    for key in keys:
        output = "{}: {}\n".format(key, metrics[key])
        string+=output
    return string
if __name__ == "__main__":
    main()
