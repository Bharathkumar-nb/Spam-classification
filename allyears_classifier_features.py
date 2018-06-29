# imports:
import time
import datetime
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
import pickle

def show_most_informative_features(feature_names, clf, n=30):
    #feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
    df = pd.DataFrame(data = coefs_with_fns, columns=['weight', 'feature'])
    output_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'Features', sys.argv[1].split('.')[0]+'.features')
    df.to_csv(output_file)

if len(sys.argv) < 3:
    print("File Usage: python allyears_classifier_features.py model_all_mails0_075.pkl\
        vectorizer_all_mails0_075.pkl\nModels are in Results/all_year_mails/")
    sys.exit(1)

model_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[1])
vectorizer_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[2])

with open(vectorizer_file, 'rb') as f:
    vocabulary = pickle.load(f)
#vectorizer = CountVectorizer(vocabulary=vocabulary)

classifier = joblib.load(model_file)

# print('Informative features')
show_most_informative_features(vocabulary, classifier)
