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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


def plot_roc(y, y_pred, file_name):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print(roc_auc_score(y, y_pred))
    # Compute micro-average ROC curve and ROC area
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.close()

ham_file = os.path.join(os.getcwd(),'Dataset', 'ham.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'spam.csv')


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


X_test = vectorizer.transform(X_test_raw)
predictions = classifier.predict(X_test)

print('accuracy_score: {}'.format(accuracy_score(predictions, y_test)))
print('zero_one_loss: {}'.format(zero_one_loss(predictions, y_test)))
print('classification_report:\n{}'.format(classification_report(predictions, y_test)))
print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, y_test)))
plot_roc(y_test, predictions, 'test.png')

# Test lingspam database

print('Lingspam')

ham_file = os.path.join(os.getcwd(),'Dataset', 'lingspam_public', 'ham.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', 'lingspam_public', 'spam.csv')

ham_df = pd.read_csv(ham_file)
ham_df['label'] = 0

spam_df = pd.read_csv(spam_file)
spam_df['label'] = 1

df = pd.concat([ham_df, spam_df])
df.dropna(inplace=True)
df = df.sample(frac=1)

X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)

print('accuracy_score: {}'.format(accuracy_score(predictions, df['label'])))
print('zero_one_loss: {}'.format(zero_one_loss(predictions, df['label'])))
print('classification_report:\n{}'.format(classification_report(predictions, df['label'])))
print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, df['label'])))
plot_roc(df['label'], predictions, 'lingspam.png')

# Test 2005_spam.csv

print('\n\n2005_spam.csv')
spam_file = os.path.join(os.getcwd(),'Dataset', '2005_spam.csv')
spam_df = pd.read_csv(spam_file)
spam_df['label'] = 1

df = pd.concat([ham_df, spam_df])
df.dropna(inplace=True)

df = df.sample(frac=1)

X_test = vectorizer.transform(df['content'])
predictions = classifier.predict(X_test)


print('accuracy_score: {}'.format(accuracy_score(predictions, df['label'])))
print('zero_one_loss: {}'.format(zero_one_loss(predictions, df['label'])))
print('classification_report:\n{}'.format(classification_report(predictions, df['label'])))
print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, df['label'])))
plot_roc(df['label'], predictions, '2005_spam.png')

