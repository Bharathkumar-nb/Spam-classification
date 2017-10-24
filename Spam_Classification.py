
# coding: utf-8

# ### Spam Classification

# #### Tasks
# 
# 1. Read files
# 2. Tokenize (Bag of words/ ngrams)
# 3. Remove stop words.
# 4. Feature hashing.
# 5. Train
# 6. Predict

# In[1]:

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import coo_matrix, hstack

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


# #### 1. Read file

# In[2]:

def get_text_from_email(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload()
            if payload != None:
                parts.append( payload )
    
    if msg["Date"]:
        e_date = email.utils.parsedate_tz(msg['Date'])
        if e_date:
            date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(e_date))
            parts.append('weekday_'+str(date.weekday())+' hour_'+str(date.hour))
    if msg['Subject']:
        parts.append(msg['Subject'])

    return ''.join(parts)


# In[3]:

ham_folder = os.path.join(os.getcwd(),'Dataset', 'Ham')
spam_folder = os.path.join(os.getcwd(),'Dataset', '2004')

ham_list = []
spam_list = []

for subdir, dirs, files in os.walk(ham_folder):
    for file in files:
        with open(os.path.join(subdir, file)) as f:
            ham_list.append(get_text_from_email(email.message_from_file(f)))

for subdir, dirs, files in os.walk(spam_folder):
    for file in files:
        try:
            with open(os.path.join(subdir, file), encoding='latin-1') as f:
                spam_list.append(get_text_from_email(email.message_from_file(f)))
        except OSError:
            pass


# In[4]:

vectorizer = CountVectorizer()


# In[5]:

len(ham_list)


# In[7]:

len(spam_list)


# In[ ]:



