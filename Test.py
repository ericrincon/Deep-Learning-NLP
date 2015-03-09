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

def transfer_learning2():
    path = 'datasets/'
    names = {1: 'title', 4: 'abstract', 5:'mesh', 'y': 6}

    #get all the data paths and shuffle them randomly
    files = [f for f in listdir(path) if isfile(join(path,f))]
    #remove the first element of files because of OSX file .DS_Store that exists in every directory
    files.pop(0)
    indices = list(range(len(files)))
    shuffle(indices)
    #get random dataset off the data list and use as held out data to train at the end
    held_out_data_path = files.pop()
    held_out_data_loader = DataLoader(path+held_out_data_path)
    held_out_data = held_out_data_loader.get_feature_matrix(names)
    #split the held out data into train and test data
    held_out_data_x = held_out_data[0]
    held_out_data_y = held_out_data[1]

    shuffled_held_out_data_train_data = held_out_data_loader.shuffle_x_and_y(held_out_data_x, held_out_data_y)
    n_documents = held_out_data_x.shape[0]
    limit = n_documents - (n_documents % 10)
    x = shuffled_held_out_data_train_data[0]
    y = shuffled_held_out_data_train_data[1]
    limit = n_documents * .8
    held_out_train_x = x[:limit]
    held_out_test_x = x[limit+1:]
    held_out_train_y = y[:limit]
    held_out_test_y = y[limit + 1:]

    #load the n-1 data
    data_loader = DataLoader(path)
    data = data_loader.get_feature_matrix({1: 'title', 4: 'abstract', 5:'mesh', 'y': 6})
    x = data[0]
    y = data[1]
    shuffled_train_data = data_loader.shuffle_x_and_y(x, y)
    n_documents = x.shape[0]
    limit = n_documents - (n_documents % 10)
    x = shuffled_train_data[0]
    y = shuffled_train_data[1]
    limit = n_documents * .8
    train_x = x[:limit]
    test_x = x[limit+1:]
    train_y = y[:limit]
    test_y = y[limit + 1:]

    mlp_train_x = train_x
    mlp_train_y = train_y

    mlp_test_x = test_x
    mlp_test_y = test_y




    #Get the feature matrix created by sklearn count vectorizer and tfidf vectorizer features should be 50K + 1
    #due to concatenated vector that corresponds to original indices of the data
    target_domain_start = train_x.shape[0] + 1
    train_x = numpy.concatenate((train_x, held_out_train_x))
    train_y = numpy.concatenate((train_y, held_out_train_y))
    data = data_loader.get_train_test_data([train_x, train_y])

    indices_target_domain = data[0][target_domain_start:, 0]
    [train_x, train_y] = data_loader.underSample(data[0], data[1])
    test_x = data_loader.transform(test_x).todense()
    indices_target_domain_exists = numpy.in1d(train_x[:, 0],indices_target_domain)
    indices_target_domain = numpy.squeeze(numpy.where(indices_target_domain_exists == True))

    target_domain_train_x = train_x[indices_target_domain,:]
    train_y_target_domain = train_y[indices_target_domain]
    train_x = numpy.delete(train_x, indices_target_domain, 0)
    train_y = numpy.delete(train_y, indices_target_domain)
    print(train_x[:, 0].shape)
    print(mlp_train_x.shape)

    mlp_train_x = mlp_train_x[train_x[:, 0].astype(int)]
    mlp_train_y = mlp_train_y[train_x[:, 0].astype(int)]

    #Save the indices for train_x
    train_x_indices = train_x[:, 1:]

    #Remove the 1st column from train_x and target_domain_train_x since it equals the indices first used
    train_x = train_x[:, 1:]
    target_domain_train_x = target_domain_train_x[:, 1:]

    #Create the augmented input space for domain adaptation
    #EXAMPLE: For a 2 domain set
    #Input function: phi(x) = <x, x, 0>
    #Target function: phi(x) = <x, 0, x>
    r_x = train_x.shape[0]
    c_x = train_x.shape[1]

    #Create N copies of data for augmented feature matrix for the source domain
    #In this case it would be 14 domain sources
    augmented_train_x = train_x
    zero_matrix_x = numpy.zeros((r_x, c_x))


    r_target_x = target_domain_train_x.shape[0]
    c_target_x = target_domain_train_x.shape[1]

    augmented_train_target_x = target_domain_train_x
    zero_matrix_target_x = numpy.zeros((r_target_x, c_target_x))

    i = 0
    r = test_x.shape[0]
    c = test_x.shape[1]
    zero_test_matrix = numpy.zeros((r, c))
    augmented_test_x = test_x
    for domain_i in range(len(files)):
        augmented_train_x = numpy.hstack((augmented_train_x, train_x))
        augmented_train_target_x = numpy.hstack((augmented_train_target_x, zero_matrix_target_x))
        augmented_test_x = numpy.hstack((augmented_test_x, zero_test_matrix))
    augmented_test_x = numpy.hstack((augmented_test_x, test_x))
    augmented_train_target_x = numpy.hstack((augmented_train_target_x, target_domain_train_x))
    augmented_train_x = numpy.hstack((augmented_train_x, zero_matrix_x))
    final_augmented_matrix = numpy.vstack((augmented_train_target_x, augmented_train_x))
    final_train_y = numpy.hstack((train_y_target_domain, train_y))
   # svm = SVM()
    #svm.train(final_augmented_matrix, final_train_y)
    #svm.test(augmented_test_x, test_y)
    #print(svm)
    #MLP train

    #pop one dataset off the data list to setup for numpy vstack()

    #transform the stacked data matrix into a feature matrix with bag of words and tfidf then undersample the negative
    #  examples
    names = {1: 'title', 4: 'abstract', 5:'mesh', 'y': 6}
    mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 75, 'batch_size': 20,
                       'activation_function': T.tanh, 'train_p':.6, 'n_layers': [2]}
    mlp_data_loader = DataLoader()
    [mlp_train_x, mlp_train_y] = mlp_data_loader.get_train_test_data([mlp_train_x, mlp_train_y], add_index_vector=False)
    mlp_train_x = mlp_train_x[:, 1:]
    print(mlp_train_x.shape)
    mlp_test_x = mlp_data_loader.transform(mlp_test_x).todense()
    classifier = NeuralNet(parameter_list=mlp_parameters)

    classifier.train(mlp_train_x, mlp_train_y)
    classifier.test(mlp_test_x, mlp_test_y)
    print(classifier)
    #get the weights and bias values of the trained MLPs hidden layer
    W = classifier.mlp.hidden_layers[0].W
    b = classifier.mlp.hidden_layers[0].b

    #Get bag of words and tfidf feature vector from held out data set
    transformed_input_train_x = data_loader.transform(held_out_train_x).todense()
    print(transformed_input_train_x.shape)
    transformed_input_train = [transformed_input_train_x, held_out_train_y]
    transformed_input_test_x = data_loader.transform(held_out_test_x).todense()
    transformed_input_test = [transformed_input_test_x, held_out_test_y]
    input_train_x = theano.sha
    #Transform the feature vectors of the held out data to the learned hidden layer features of the previous
    #MLP trained with all n-1 datasets
    a_function = mlp_parameters['activation_function']
    input_train = a_function(T.dot(transformed_input_train[0], W) + b)
    input_test = a_function(T.dot(transformed_input_test[0], W) + b)

    input_train = transformed_input_train[:3360, :]

    #set up and train the mlp on the held out dataset
    mlp_parameters = {"learning_rate": 1, 'L1_term': 0.000, 'L2_term': .001, 'n_epochs': 75, 'batch_size': 20,
                       'activation_function': T.tanh, 'train_p':.6, 'n_layers': [1000]}
    transfer_mlp = NeuralNet(parameter_list=mlp_parameters)
    transfer_mlp.train(input_train, held_out_train_y)
    transfer_mlp.test(input_test[:, 1:], held_out_test_y)
    print(transfer_mlp)

def transfer_learning():
    path = 'datasets/'
    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}

    #get all the data paths and shuffle them randomly
    files = [f for f in listdir(path) if isfile(join(path,f))]
    #remove the first element of files because of OSX file .DS_Store that exists in every directory
    files.pop(0)

    #Shuffle the files
    shuffle(files)

    data_loader = DataLoader(path)

    #Get one file out of the csv files in the dataloader use this as the held out domain
    held_out_domain = data_loader.csv_files.pop()
    held_out_x, held_out_y = data_loader.get_feature_matrix(names, held_out_domain)
    held_out_x, held_out_y = data_loader.get_train_test_data([held_out_x, held_out_y], True, False)

    #Get the total number of domains i.e., the number of files with documents
    n_source_domains = len(files)

    #Read the n-1 csv files
    source_x, source_y = data_loader.get_feature_matrix(names)
    source_x = data_loader.transform(source_x, True)
    previous_domain_sizes = 0

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


    target_source = held_out_x
    target_source_zero_matrix = numpy.zeros((target_source.shape[0], target_source.shape[1]))

    for i in range(len(files) - 1):
        if not i == (len(files) - 1):
            target_source = scipy.sparse.hstack((target_source, target_source_zero_matrix))
        else:
            target_source = scipy.sparse.hstack((target_source, held_out_x))

    target_split = round(held_out_y.shape[0] * .7)
    train_target_x = target_source.tocsr()[:target_split, :]

    index_stop = source_domain_x.shape[0] - (source_domain_x.shape[0] % 10)

    mlp_train_x = source_domain_x.tocsr()[:index_stop, :]

    mlp_train_x = mlp_train_x.tocsc()[:, :10000].todense()
    mlp_train_y = source_domain_y[:index_stop]

    train_x = scipy.sparse.vstack((source_domain_x, train_target_x))
    train_y = numpy.hstack((source_domain_y, held_out_y[:target_split]))

    test_x = target_source.tocsr()[target_split + 1:, :]
    test_y = held_out_y[target_split + 1:]

    perceptron_train_x = train_target_x.tocsc()[:, :10000].todense()
    perceptron_train_y = held_out_y[:target_split]

    perceptron_test_x = target_source.tocsr()[target_split + 1:, :]
    perceptron_test_x = perceptron_test_x.tocsc()[:, :10000].todense()
    perceptron_test_y = held_out_y[target_split + 1:]

    svm = SVM()
    svm.train(train_x, train_y)
    svm.test(test_x, test_y)
    print(svm)

    mlp_parameters = {"learning_rate": 1.2, 'L1_term': 0.000, 'L2_term': .0001, 'n_epochs': 100, 'batch_size': 20,
                       'activation_function': T.tanh, 'train_p':.6, 'n_layers': [1500]}

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


    perceptron = Perceptron(penalty="l2", n_iter=100)
    perceptron.fit(transformed_perceptron_train_x, perceptron_train_y)
    prediction = perceptron.predict(transformed_perceptron_test_x)
    scores = []
    scores.append(f1_score(perceptron_test_y, prediction))
    scores.append(precision_score(perceptron_test_y, prediction))
    scores.append(recall_score(perceptron_test_y, prediction))
    scores.append(roc_auc_score(perceptron_test_y, prediction))
    scores.append(accuracy_score(perceptron_test_y, prediction))

    print("perceptron scores")
    for score in scores:
        print(score)
if __name__ == "__main__":
    main()
