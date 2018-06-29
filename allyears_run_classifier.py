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
from sklearn.metrics import confusion_matrix, accuracy_score, zero_one_loss, classification_report, roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
import pickle

if len(sys.argv) < 4:
	print('File usage: python allyears_run_classifier.py model3.pkl vectorizer3.pkl spam_file\nModel is in Results/all_year_mails/')
	sys.exit(1)

model_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[1])
vectorizer_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[2])
classifier = joblib.load(model_file)

ham_file = os.path.join(os.getcwd(),'Dataset', 'Test', 'ham.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'Test', sys.argv[3])


ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df['label'] = 0
ham_df.dropna(inplace=True)
ham_df = ham_df.sample(frac=0.2)

spam_df = pd.read_csv(spam_file, encoding='latin-1')
spam_df['label'] = 1
spam_df.dropna(inplace=True)
spam_df = spam_df.sample(frac=0.2)

df = pd.concat([ham_df, spam_df])
df = df.sample(frac=1)

with open(vectorizer_file, 'rb') as f:
    vocabulary = pickle.load(f)
vectorizer = CountVectorizer(vocabulary=vocabulary)

X_test =  vectorizer.transform(df['content'])

predictions = classifier.predict(X_test)

print('accuracy_score: {}'.format(accuracy_score(predictions, df['label'])))
print('zero_one_loss: {}'.format(zero_one_loss(predictions, df['label'])))
print('classification_report:\n{}'.format(classification_report(predictions, df['label'])))
print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, df['label'])))
