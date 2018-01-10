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
from scipy.sparse import coo_matrix, hstack

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

def show_most_informative_features(vectorizer, clf, n=30):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    # top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    # for (coef_1, fn_1), (coef_2, fn_2) in top:
    #     print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

    for (coef_1, fn_1) in coefs_with_fns[:n]:
        print('{} {}'.format(coef_1, fn_1))

    for (coef_1, fn_1) in coefs_with_fns[:-(n+1):-1]:
        print('{} {}'.format(coef_1, fn_1))

ham_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', 'ham.csv')
ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df.dropna(inplace=True)
ham_df['label'] = 0

for i in range(1,len(sys.argv)):

    spam_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', sys.argv[i])
    print('file: {}'.format(sys.argv[i]))
    spam_df = pd.read_csv(spam_file, encoding='latin-1')
    spam_df.dropna(inplace=True)
    spam_df = spam_df.sample(frac=0.5)
    spam_df['label'] = 1
    df = pd.concat([spam_df, ham_df])
    del spam_df

    df = df.sample(frac=1)

    additional_stop_words = ['enron','vince','louise','attached','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']

    vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1, 2))
    y_train = df['label']
    X_train =  vectorizer.fit_transform(df['content'])
    del df 
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    del X_train, y_train

    # print('Informative features')
    show_most_informative_features(vectorizer, classifier)
    # print('\n---------------------------------\n')

    # print('Top 20')
    # print_top20(vectorizer, classifier) 
