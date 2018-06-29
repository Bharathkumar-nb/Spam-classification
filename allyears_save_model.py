# imports:
import time
import datetime
import os
import sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
import pickle

if len(sys.argv) < 2:
	print('File usage: python allyears_save_model.py all_mails2.csv\ncsv is in Dataset/all_year_mails_csv/')
	sys.exit(1)


spam_file = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', sys.argv[1])
model_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'model_' +sys.argv[1].split('.')[0]+'.pkl')
vectorizer_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'vectorizer_' +sys.argv[1].split('.')[0]+'.pkl')
ham_file = os.path.join(os.getcwd(),'Dataset', 'csv_files', 'ham.csv')

ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df.dropna(inplace=True)
ham_df['label'] = 0

spam_df = pd.read_csv(spam_file, encoding='latin-1')
df = pd.concat([spam_df, ham_df])
del spam_df


additional_stop_words = ['enron','vince','louise','attached','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
additional_stop_words += ['koi8', 'http', 'windows', 'utf', 'nbsp', 'bruceg']
#vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1, 2))
vectorizer = CountVectorizer(min_df=3, stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1, 2))
X_train =  vectorizer.fit_transform(df['content'])
del df['content']
print("Created X_train")
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer.get_feature_names(), f)
#joblib.dump(vectorizer, vectorizer_file)
print("Dumped")
print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[:50])
classifier = LogisticRegression()
classifier.fit(X_train, df['label'])
del X_train


joblib.dump(classifier, model_file)