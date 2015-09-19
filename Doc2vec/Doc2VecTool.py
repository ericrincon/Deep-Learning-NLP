__author__ = 'ericrincon'

__author__ = 'ericrincon'

import os
import gensim
import numpy
import linecache

from random import shuffle
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import gensim.models.doc2vec
import multiprocessing

n_documents = 1104935

class LabeledLineDocument(object):
    def __init__(self, source):
        #Create a list as big as the number of lines in file from 1 to n that is permuted
        self.source = source
    def __iter__(self):
        n_random_nums = numpy.random.permutation(range(1, n_documents + 1))

        with utils.smart_open(self.source) as file:
            for line_number, line in zip(n_random_nums, file):
                yield TaggedDocument(utils.to_unicode(line).split(), [line_number])
    def to_array(self):

        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [item_no]))
        return self.sentences

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



"""
    Simple method for finding and returning all file directoies in root and subroot directory.
"""
def get_all_files(path):
    file_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:
            #Make sure hidden files do not make into the list
            if name[0] == '.':
                continue
            file_paths.append(os.path.join(path, name))
    return file_paths





def preprocess_line(line):
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



def preprocess():
    pubmed_dir = '/Users/ericrincon/Documents/Pubmed/Text/'
    pubmed_folders = os.listdir(pubmed_dir)
    pubmed_folders.pop(0)
    punctuation = "`~!@#$%^&*()_-=+[]{}\|;:'\"|<>,./?åαβ"
    numbers = "1234567890"
    number_replacement = create_whitespace(len(numbers))
    spacing = create_whitespace(len(punctuation))

    files = get_all_files(pubmed_dir)

    output_file = open('preprocessed_text.txt', 'w')

    for i, file in enumerate(files):
        tokens = set([])
        text_file_object = open(file, 'r')


        for line in text_file_object:
            if 'Open Access' in line:
                break
            else:
                if line.strip() == '':
                    continue
                lowercase_line = line.lower()
                translation_table = str.maketrans(punctuation, spacing)
                translated_line = lowercase_line.translate(translation_table)
                translation_table_numbers = str.maketrans(numbers, number_replacement)
                final_line = translated_line.translate(translation_table_numbers)
                line_tokens = utils.to_unicode(final_line).split()
                tokens = list(set(tokens) | set(line_tokens))

                preprocessed_text = ''

        #Create string with all tokens to write to file
        for token in tokens:
            preprocessed_text = preprocessed_text + ' ' + token

        output_file.write(preprocessed_text + '\n')
        print('Document written: ', i)

def main():
    #preprocess()
    start_training()

def start_training():
    cores = multiprocessing.cpu_count()
    epochs = 20
    """
    Maybe try alpha tuning later
    alpha = 0.025
    min_alpha =  0.001
    alpha_delta = (alpha - min_alpha) / epochs
    """
    #Read preprocessed text file with gensim TaggedLineDocument
    documents = LabeledLineDocument('preprocessed_text.txt')
    model = Doc2Vec(documents, size=400, window=10, min_count=5, workers=cores)

    model.build_vocab(documents)


    for epoch in range(epochs):
        #shuffle the documents every epoch to get better results
        model.train(documents)



    model.save('400_pvdm_doc2vec.d2v')



if __name__ == '__main__':
    main()