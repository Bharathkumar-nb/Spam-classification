import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
import pickle

if len(sys.argv)<2:
    print("File Usage: python synthetic_attack_run.py all_mails0_075\
        \nInput folder is in Dataset/all_year_mails_csv/")
    sys.exit(1)

print(sys.argv[1])
dataset_folder = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', sys.argv[1])
vectorizer_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', "vectorizer_{}.pkl".format(sys.argv[1]))
model_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', "model_{}.pkl".format(sys.argv[1]))

with open(vectorizer_file, 'rb') as f:
    vocabulary = pickle.load(f)

vectorizer = CountVectorizer(vocabulary=vocabulary)

spam_file = os.path.join(os.getcwd(),'Dataset', 'Test', sys.argv[2])
spam_df = pd.read_csv(spam_file, encoding='latin-1')
X_test =  vectorizer.transform(spam_df['content'])
print((X_test).max(1))
y = [x.item(0,0) for x in (X_test!=0).sum(1)]
#y = [min(100,x.item(0,0)) for x in (X_test!=0).sum(1)]
#y = [min(50,x.item(0,0)) for x in (X_test!=0).sum(1)]
y = [min(50,x.item(0,0)) for x in X_test.sum(1)]
print(y[:10])
plt.plot(sorted(y), lw=2)
plt.xlabel("Mails")
plt.ylabel("Number of words in mail")
plt.savefig('tmp3.png')
plt.close()
# classifier = joblib.load(model_file)

# print(len(vocabulary))
# TOTAL_COST = 10
# cost = np.random.randint(1, high=3, size=len(vocabulary))
# print(cost)
# print(classifier.coef_[0])
# print(classifier.coef_[0]/cost)