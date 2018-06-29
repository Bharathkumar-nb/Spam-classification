# imports:
import time
import datetime
import os
import sys
import pandas as pd
#from sklearn.utils import shuffle

yearly_spam_folder = os.path.join(os.getcwd(),'Dataset', 'csv_files')

for subdir, dirs, files in os.walk(yearly_spam_folder):
    if dirs != []:
        continue
    for spam_file in files:
        if 'spam1' not in spam_file and 'spam2' not in spam_file:
            target_train_file = os.path.join(os.getcwd(),'Dataset', 'Train', spam_file)
            target_test_file = os.path.join(os.getcwd(),'Dataset', 'Test', spam_file)
            print('file: {}'.format(spam_file))
            df = pd.read_csv(os.path.join(subdir, spam_file), encoding='latin-1')
            df.dropna(inplace=True)
            #df = shuffle(df)
            df = df.sample(frac=1)
            train_no = int(0.8*df.shape[0])
            pd.DataFrame(df[:train_no], columns=['content']).to_csv(target_train_file)
            pd.DataFrame(df[train_no:], columns=['content']).to_csv(target_test_file)