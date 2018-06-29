import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS

spam_file = os.path.join(os.getcwd(),'Dataset', 'monthly_dataset_csv', sys.argv[1])

spam_df = pd.read_csv(spam_file, encoding='latin-1')

additional_stop_words = ['enron','vince','louise','attached','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
additional_stop_words += ['koi8', 'http', 'windows', 'utf', 'nbsp', 'bruceg']
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union(additional_stop_words))

#print(spam_df['content'].head())

X_train = vectorizer.fit_transform(spam_df['content'])
print(X_train.size)
print(spam_df.shape)
#print(vectorizer.get_feature_names()[2591])
#print(vectorizer.get_feature_names()[2590])
#print(vectorizer.get_feature_names()[2592])
#print(spam_df['content'][0])
#print(spam_df['content'][2])
#print(np.max(X_train, axis=1))
print(np.sum(X_train, axis=1))
print(X_train)
#print(np.bincount(np.max(X_train, axis=1)))