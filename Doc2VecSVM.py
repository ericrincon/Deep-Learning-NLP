__author__ = 'ericrincon'

from gensim.models import Doc2Vec
from NeuralNet import NeuralNet
from os import listdir
from os.path import isfile, join
from DataLoader import DataLoader
from gensim import utils
from SVM import SVM
import sys
import numpy
import Doc2vec.Doc2VecTool
import scipy.sparse as sparse

def main():
    dm_model = Doc2Vec.load('400_pvdm_doc2vec.d2v')
    dbow_model = Doc2Vec.load('400_pvdbow_doc2vec.d2v')

    #Load datasets for classfying
    path = 'datasets/'
    doc2vec_vector_size = 400
    files = [f for f in listdir(path) if isfile(join(path,f))]
    files.pop(0)

    data_loader = DataLoader(path)

    domains = data_loader.csv_files


    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}

    domain_features = data_loader.get_feature_matrix(names)
    domain = domain_features.pop(0)
    x, y = domain
    #get size
    n_total_documents = 0

    for domain in domain_features:
        n_total_documents+=len(domain[0])
        x = numpy.hstack((x, domain[0]))
        y = numpy.hstack((y, domain[1]))
    x, y = data_loader.create_random_samples(x, y, train_p=.8, test_p=.2)
    train_x, test_x = x
    train_y, test_y = y
    transformed_train_x = data_loader.get_transformed_features(train_x, sparse=True, tfidf=True, add_index_vector=False)
    transformed_test_x = data_loader.get_transformed_features(test_x, sparse=True, tfidf=True)
    all_features = numpy.zeros(shape=(n_total_documents, 800))
    all_labels = numpy.asarray([])

    i = 0

    dbow_dm_train_x = numpy.zeros((train_x.shape[0], 2*doc2vec_vector_size))
    dbow_dm_test_x = numpy.zeros((test_x.shape[0], 2*doc2vec_vector_size))

    """
        Set up the feature for the SVM by iterating through all the word vectors.
        Pre process each vector and then feed into doc2vec model, both the distributed memory
        and distributed bag of words. Concatenate the vectors for better classification results
        as per paragraph to vector paper by Mikolv.
    """
    for feature_vector in train_x:
        preprocessed_line = list(Doc2vec.Doc2VecTool.preprocess_line(feature_vector))
        dbow_dm_train_x[i, 0:400] = dm_model.infer_vector(preprocessed_line)
        dbow_dm_train_x[i, 400:] = dbow_model.infer_vector(preprocessed_line)
        i+=1

    """
        Do the same as above but for the test set.
    """

    i = 0

    for feature_vector in test_y:
        preprocessed_line = list(Doc2vec.Doc2VecTool.preprocess_line(feature_vector))
        dbow_dm_test_x[i, 0:400] = dm_model.infer_vector(preprocessed_line)
        dbow_dm_test_x[i, 400:] = dbow_model.infer_vector(preprocessed_line)
        i+=1

    print("Training doc2vec SVM")
    #Train SVM on classic bow
    svm = SVM()
    svm.train(dbow_dm_train_x, train_y)
    svm.test(dbow_dm_test_x, test_y)
    print("end of training doc2vec bow SVM\n")


    print("Training classic bow SVM")
    #Train SVM on classic bow
    svm = SVM()
    svm.train(transformed_train_x, train_y)
    svm.test(transformed_test_x, test_y)
    print("end of training classic bow SVM\n")

    #

if __name__ == '__main__':
    main()
