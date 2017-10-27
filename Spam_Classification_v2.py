# imports:
import time;
import datetime
import logging
import os
import numpy as np
import pandas as pd
import re
import random
import email
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics 
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, zero_one_loss, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import coo_matrix, hstack

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO



ham_file = os.path.join(os.getcwd(),'Dataset', 'ham_latin.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'spam_latin.csv')


# ham_df = pd.read_csv(ham_file)
ham_df = pd.read_csv(ham_file, encoding='latin-1')
ham_df['label'] = 0

# spam_df = pd.read_csv(spam_file)
spam_df = pd.read_csv(spam_file, encoding='latin-1')
spam_df['label'] = 1

df = pd.concat([ham_df, spam_df])

df.dropna(inplace=True)

df = df.sample(frac=1)

# print(df)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['content'],df['label'])

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)

print('Training')

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# X_test = vectorizer.transform(X_test_raw)
# predictions = classifier.predict(X_test)

# print(list(zip(predictions, y_test))[:20])
# print(accuracy_score(predictions, y_test))
# print(zero_one_loss(predictions, y_test))


ham_file = os.path.join(os.getcwd(),'Dataset', 'lingspam_public', 'ham.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'lingspam_public', 'spam.csv')


ham_df = pd.read_csv(ham_file)
ham_df['label'] = 0

spam_df = pd.read_csv(spam_file)
spam_df['label'] = 1

df = pd.concat([ham_df, spam_df])

df.dropna(inplace=True)

df = df.sample(frac=1)

# print(df)

print('Lingspam')
X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)

# print(list(zip(predictions, df['label']))[:20])
print(accuracy_score(predictions, df['label']))
print(zero_one_loss(predictions, df['label']))


print('2005_spam.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', '2005_spam.csv')
df = pd.read_csv(spam_file)
df['label'] = 1
df.dropna(inplace=True)

# print(df)

X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)

# print(list(zip(predictions, df['label']))[:20])
print(accuracy_score(predictions, df['label']))
print(zero_one_loss(predictions, df['label']))

print('2006_spam.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', '2006_spam.csv')
df = pd.read_csv(spam_file)
df['label'] = 1
df.dropna(inplace=True)

# print(df)

X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)

# print(list(zip(predictions, df['label']))[:20])
print(accuracy_score(predictions, df['label']))
print(zero_one_loss(predictions, df['label']))

print('2010_spam.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', '2010_spam.csv')
df = pd.read_csv(spam_file)
df['label'] = 1
df.dropna(inplace=True)

# print(df)

X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)

# print(list(zip(predictions, df['label']))[:20])
print(accuracy_score(predictions, df['label']))
print(zero_one_loss(predictions, df['label']))
