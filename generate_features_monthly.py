# imports:
import time
import datetime
import os
import sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 


def show_most_informative_features(vectorizer, clf, spam_file, n=30):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    # top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    # for (coef_1, fn_1), (coef_2, fn_2) in top:
    #     print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    df = pd.DataFrame(data = coefs_with_fns, columns=['weight', 'feature'])
    df['file'] = spam_file
    output_file = spam_file.split('.')[0] + '_features.out'
    df.to_csv(os.path.join(os.getcwd(),'Results','features','1gram',output_file))

ham_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', 'ham.csv')
ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df.dropna(inplace=True)
ham_df['label'] = 0

monthly_spam_folder = os.path.join(os.getcwd(),'Dataset', 'monthly_dataset_csv')

for subdir, dirs, files in os.walk(monthly_spam_folder):
    if dirs != []:
        continue
    for spam_file in files:
        print('file: {}'.format(spam_file))
        spam_df = pd.read_csv(os.path.join(subdir, spam_file), encoding='latin-1')
        spam_df.dropna(inplace=True)
        spam_df['label'] = 1
        df = pd.concat([spam_df, ham_df])
        del spam_df

        df = df.sample(frac=1)

        additional_stop_words = ['enron','vince','louise','attached','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
        additional_stop_words += ['koi8', 'http', 'windows', 'utf', 'nbsp', 'bruceg']
        more_words_file = open(os.path.join(os.getcwd(),'Results','features','removed_features_list.dmp'))
        more_words = more_words_file.readlines()
        more_words_file.close()
        additional_stop_words += more_words
        start_time = time.time()
        #vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1, 2))
        vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words))
        y_train = df['label']
        X_train =  vectorizer.fit_transform(df['content'])
        del df 
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        print('Training time = {}'.format(time.time() - start_time))
        del X_train, y_train
        
        # print('Informative features')
        show_most_informative_features(vectorizer, classifier, spam_file)
