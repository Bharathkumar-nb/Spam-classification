#############################################################
#### NOTE: Do not change print statements                ####
#### NOTE: Output of this file is used by other programs ####
#############################################################
# imports:
import time;
import datetime
import logging
import os
import sys
import numpy as np
import pandas as pd
import re
import random
import email
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics 
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, zero_one_loss, classification_report, roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

# Check command line arguments
if len(sys.argv) < 2:
    print('File Usage: python classifier.py <train-file> ...<optional-test-files>')
    print('Location of <files>: Dataset/csv_files/')
    sys.exit(0)

def print_top20(vectorizer, clf):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    top20 = np.argsort(clf.coef_[0])[-20:]
    print("%s" % ("\n".join(feature_names[j] for j in top20)))

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

ham_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', 'ham.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', sys.argv[1])


ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df['label'] = 0

ham_df.dropna(inplace=True)

ham_X_train_raw, ham_X_test_raw, ham_y_train, ham_y_test = train_test_split(ham_df['content'],ham_df['label'])

del ham_df

spam_df = pd.read_csv(spam_file, encoding='latin-1')
spam_df['label'] = 1

spam_df.dropna(inplace=True)

spam_X_train_raw, spam_X_test_raw, spam_y_train, spam_y_test = train_test_split(spam_df['content'],spam_df['label'])

del spam_df

X_train_raw = pd.concat([ham_X_train_raw, spam_X_train_raw])
del ham_X_train_raw, spam_X_train_raw
X_test_raw = pd.concat([ham_X_test_raw, spam_X_test_raw])
y_train = pd.concat([ham_y_train, spam_y_train])
del ham_y_train, spam_y_train
y_test = pd.concat([ham_y_test, spam_y_test])


df = pd.concat([X_train_raw, y_train], axis=1)
df = df.sample(frac=1)

X_train_raw = df['content']
y_train = df['label']

additional_stop_words = ['enron','vince','louise','attached','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
additional_stop_words += ['koi8', 'http', 'windows', 'utf', 'nbsp']

vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)

print('Started training')

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print('Finished training\n')

del X_train_raw, X_train, y_train

print('Informative features')
show_most_informative_features(vectorizer, classifier)

print('\n---------------------------------\n')

print('Top 20')
print_top20(vectorizer, classifier) 

print('\n---------------------------------\n')

print()
print('Prediction on heldout Test set\n')

X_test = vectorizer.transform(X_test_raw)
predictions = classifier.predict(X_test)

predictions_prob = classifier.predict_proba(X_test)

print('accuracy_score: {}'.format(accuracy_score(predictions, y_test)))
print('zero_one_loss: {}'.format(zero_one_loss(predictions, y_test)))
print('classification_report:\n{}'.format(classification_report(predictions, y_test)))
print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, y_test)))

filepath = os.path.join(os.getcwd(),'Results',sys.argv[1].split('.')[0]+'_test.png')
plot_roc(y_test, predictions, predictions_prob, filepath)

print('\n\n---------------------------------\n\n')

for i in range(2, len(sys.argv)):
    graph_filename = sys.argv[1].split('.')[0] + '_on_' + sys.argv[i].split('.')[0]
    print (graph_filename+'\n')

    spam_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', sys.argv[i])
    spam_df = pd.read_csv(spam_file)
    spam_df['label'] = 1
    spam_df.dropna(inplace=True)
    
    X_test_raw = pd.concat([ham_X_test_raw, spam_df['content']])
    y_test = pd.concat([ham_y_test, spam_df['label']])
    del spam_df
    X_test = vectorizer.transform(X_test_raw)
    del X_test_raw
    predictions = classifier.predict(X_test)
    predictions_prob = classifier.predict_proba(X_test)
    
    print('accuracy_score: {}'.format(accuracy_score(predictions, y_test)))
    print('zero_one_loss: {}'.format(zero_one_loss(predictions, y_test)))
    print('classification_report:\n{}'.format(classification_report(predictions, y_test)))
    print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, y_test)))

    filepath = os.path.join(os.getcwd(),'Results',graph_filename+'.png')
    plot_roc(y_test, predictions, predictions_prob, filepath)
    print('\n\n---------------------------------\n\n')
