__author__ = 'ericRincon'

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'] #Categorie list for the dataset
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42) #Get training dataset based on categories
#print("\n".join(twenty_train.data[0].split("\n")[:3]))

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

#Tokenize text to build a dictionary of features and transform documents to feature vectors
feature_vector = count_vect.vocabulary_.get(u'algorithm')

#Longer Documents will have a higher average count value than shorted documents.
#So, to avoid potential discrepancies we divide the number of occurrences of each word by the total number of words in
#the document. We call these new features Term frequencies

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X__train_tf = tf_transformer.transform(X_train_counts)

#With all the features created we can train the classifier

clf = MultinomialNB().fit(X__train_tf, twenty_train.target)

#Pipeline class which meakes it easier to modify data and the such
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
#Get the test data to test predication accuracy rate
twenty_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
percent_correct = np.mean(predicted == twenty_test.target)
print('Naive Bayes Classifier: ')
print(percent_correct*100)
print("\n")
#Try a support vector machine to try and beat the naive_bayes classifier

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5))])
fitted = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
percent_correct = np.mean(predicted == twenty_test.target)
print("SVM: ")
print(percent_correct * 100)
print(metrics.classification_report(twenty_test.target, predicted, target_names = twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))

parameters = {'vect_ngram_range': [(1,1), (1,2)], 'tfidf_use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs = -1)

gs_clf = gs_clf.fit(twenty_train[:400], twenty_train.target[:400])