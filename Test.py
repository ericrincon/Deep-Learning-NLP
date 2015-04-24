__author__ = 'eric_rincon'

import numpy
import theano
import copy

from Feature_Engineer import Feature_Engineer
from sklearn.linear_model import Perceptron

from os import listdir
from os.path import isfile, join
from DataLoader import DataLoader
from NeuralNet import NeuralNet
from SVM import SVM

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from scipy import sparse

def main():
    transfer_learning()

def transfer_learning():
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
    all_domains_perceptron_metrics_list = []
    all_domains_perceptron_modified_metrics_list = []
    all_domains_perceptron_without_list = []
    all_domains_svm_bow_mlp_list = []
    all_domains_mlp_fold_scores = []
    all_domains_mlptanh_fold_scores = []

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
        mlptanh_fold_scores = []

        output = "Dataset: {}".format(files[i])
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

            classifier = NeuralNet(n_hidden_units=[250], output_size=2, batch_size=20, n_epochs=200, dropout=True,
                                   activation_function='relu', learning_rate=.3, momentum=True, momentum_term=.5)
            classifier_tanh = NeuralNet(n_hidden_units=[250], output_size=2, batch_size=20, n_epochs=200, dropout=True,
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

            perceptron = Perceptron(penalty="l2", n_iter=100)
            perceptron.fit(transformed_perceptron_train_x, neural_net_y_train)
            prediction = perceptron.predict(transformed_perceptron_test_x)
            output = "\nSVM with BoW and transformed features"
            print(output)
            svm_mlp_bow = SVM()
            svm_mlp_bow.train(sparse.coo_matrix(modified_transformed_perceptron_train_x), neural_net_y_train)
            svm_mlp_bow.test(sparse.coo_matrix(modified_transformed_perceptron_test_x), test_y)
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
            output = "*********** End of fold {} ***********".format(n_fold)
            print(output)
            file.write(output)

        training_domains = copy.deepcopy(all_domains)

        output = "----------------------------------------------------------------------------------------" \
                 "\nFold Scores\n " \
                 "SVM with domain adaptation"
        print_write_output(output, svm_wda_metrics_list, all_domains_svm_wda_metrics_list, file)

        output = "\nSVM without domain adaptation"
        print_write_output(output, svm_metrics_list, all_domains_svm_metrics_list, file)

        output = "\nPerceptron without transfer learning"
        print_write_output(output, perceptron_without_list, all_domains_perceptron_without_list, file)

        output = "\nPerceptron with transfer learning"
        print_write_output(output, perceptron_metrics_list, all_domains_perceptron_metrics_list, file)

        output = "\nPerceptron with transfer learning and concatenated bag of words"
        print_write_output(output, perceptron_modified_metrics_list, all_domains_perceptron_modified_metrics_list, file)

        output = "SVM with BoW and transformed features"
        print_write_output(output, svm_bow_mlp_list, all_domains_svm_bow_mlp_list, file)

        output = "MLP scores with relu activation function"
        print_write_output(output, mlp_fold_scores, all_domains_mlp_fold_scores, file)

        output = "\nMLP with tanh activation function"
        print_write_output(output, mlptanh_fold_scores, all_domains_mlptanh_fold_scores, file)

    output = "*******************************************************************************************" \
             "\nAll domain macro metric scores\n " \
             "SVM with domain adaptation"
    print_macro_scores(output, all_domains_svm_wda_metrics_list, file)

    output = "\nSVM without domain adaptation"
    print_macro_scores(output, all_domains_svm_metrics_list, file)

    output = "\nPerceptron without transfer learning"
    print_macro_scores(output, all_domains_perceptron_without_list, file)

    output = "\nPerceptron with transfer learning"
    print_macro_scores(output, all_domains_perceptron_metrics_list, file)

    output = "\nPerceptron with transfer learning and concatenated bag of words"
    print_macro_scores(output, all_domains_perceptron_modified_metrics_list, file)

    output = "SVM with BoW and transformed features"
    print_macro_scores(output, all_domains_svm_bow_mlp_list, file)

    output = "MLP scores with relu activation function"
    print_macro_scores(output, all_domains_mlp_fold_scores, file)

    output = "\nMLP with tanh activation function"
    print_macro_scores(output, all_domains_mlptanh_fold_scores, file)




def print_write_output(title, classifier_metrics, all_domain_metrics, file):
    file.write(title)
    print(title)
    metrics = calculate_fold_scores(classifier_metrics)
    all_domain_metrics.append(metrics)
    output_classifer = print_classifier_scores(metrics)
    print(output_classifer)
    file.write(output_classifer)
def print_macro_scores(output, classifier_metrics, file):
    file.write(output)
    print(output)
    metrics = calculate_fold_scores(classifier_metrics)
    output_classifer = print_classifier_scores(metrics)
    print(output_classifer)
    file.write(output_classifer)

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
if __name__ == "__main__":
    main()
