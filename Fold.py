__author__ = 'eric_rincon'

import threading

class Fold(threading.Thread):
    def __init__(self,fold_n, train_data, test_data, classifier, output_file):
        threading.Thread.__init__(self)
        self.train_x = train_data[0]
        self.train_y = train_data[1]
        self.test_x = test_data[0]
        self.test_y = test_data[1]
        self.classifier = classifier
        self.file = output_file
        self.n = fold_n
    def run(self):
        #[:,1:] is used because the first row of the matrix contains
        self.classifier.train(self.train_x[:, 1:], self.train_y)
        self.classifier.test(self.test_x[:, 1:], self.test_y)

        #Each classifier when outputed in string outputs their
        #respective metrics
        self.file.write(self.__str__())
    def __str__(self):
        output =  "Fold: {}\nTrain data indices: {}\n" \
               "Test data indices: {}\n" \
               "{}\n".format(self.n,
                         self.train_x[:, 0],
                         self.test_x[:, 0],
                         self.classifier)
        return output
