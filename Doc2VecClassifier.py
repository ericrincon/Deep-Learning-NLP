__author__ = 'ericrincon'

from gensim.models import Doc2Vec
from NeuralNet import NeuralNet
from os import listdir
from os.path import isfile, join
from DataLoader import DataLoader
from gensim import utils
import sys
import numpy

"""
    Create an empty string of parameter length.
    Used in conjunction with python's translation method to, in this case, pre-process text
    for Doc2Vec.
"""
def create_whitespace(length):
    whitespace = ''

    for i in range(length):
        whitespace += ' '

    return whitespace


def preprocess(line):
    punctuation = "`~!@#$%^&*()_-=+[]{}\|;:'\"|<>,./?åαβ"
    numbers = "1234567890"
    number_replacement = create_whitespace(len(numbers))
    spacing = create_whitespace(len(punctuation))

    lowercase_line = line.lower()
    translation_table = str.maketrans(punctuation, spacing)
    translated_line = lowercase_line.translate(translation_table)
    translation_table_numbers = str.maketrans(numbers, number_replacement)
    final_line = translated_line.translate(translation_table_numbers)
    line_tokens = utils.to_unicode(final_line).split()

    return set(line_tokens)

def main():
    model = Doc2Vec.load('400_pvdm_doc2vec.d2v')
    model_dbow = Doc2Vec.load('400_pvdbow_doc2vec.d2v')
    #mistake pvdm is actually pv-dbow
    path = 'datasets/'

    files = [f for f in listdir(path) if isfile(join(path,f))]
    files.pop(0)

    data_loader = DataLoader(path)

    domains = data_loader.csv_files


    names = {1: 'title', 4: 'abstract', 5: 'mesh', 'y': 6}

    domain_features = data_loader.get_feature_matrix(names)

    #get size
    n_total_documents = 0

    for domain in domain_features:
        n_total_documents+=len(domain[0])

    all_features = numpy.zeros(shape=(n_total_documents, 800))
    all_labels = numpy.asarray([])
    i = 0

    for domain in domain_features:
        features, labels = domain
        all_labels = numpy.hstack((all_labels, labels))
        for feature_vector in features:
            preprocessed_line = list(preprocess(feature_vector))
            all_features[i, 0:400] = numpy.float_(model.infer_vector(preprocessed_line))
            all_features[i, 400:] = numpy.float_(model_dbow.infer_vector(preprocessed_line))
            i+=1
    all_labels = numpy.asarray(all_labels)
    all_labels[all_labels == -1] = 0
    all_labels = numpy.intc(all_labels)
    train, test = data_loader.create_random_samples(all_features, all_labels)
    train_x, train_y = train
    test_x, test_y = test

    classifier = NeuralNet(n_hidden_units=[200], output_size=2, batch_size=20, n_epochs=200, dropout=True,
                                   activation_function='relu', learning_rate=.3, momentum=True, momentum_term=.5)

    classifier.train(train_x, train_y)
    classifier.test(test_x, test_y)

if __name__ == '__main__':
    main()