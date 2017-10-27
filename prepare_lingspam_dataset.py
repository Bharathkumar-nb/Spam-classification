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

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def get_text_from_email(msg, file_name):
    parts = {}
    parts['content'] = ' '
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload()
            if payload != None:
                # parts['content'] = payload
                parts['content'] += payload
    
    if msg["Date"]:
        e_date = email.utils.parsedate_tz(msg['Date'])
        if e_date:
            date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(e_date))
            # parts['week_day'] = str(date.weekday())
            # parts['hour_of_day'] = str(date.hour)
            parts['content'] += ' weekday_' +  str(date.weekday()) + ' hour_of_day_' + str(date.hour)
    if msg['Subject']:
        # parts['subject'] = msg['Subject']
        parts['content'] += ' ' + msg['Subject']

    return parts


mail_folder = os.path.join(os.getcwd(),'Dataset', 'lingspam_public','bare')

ham_list = []
spam_list = []

for subdir, dirs, files in os.walk(mail_folder):
    for file_name in files:
        try:
            with open(os.path.join(subdir, file_name)) as f:
                if 'spm' in file_name:
                    spam_list.append(get_text_from_email(email.message_from_file(f), file_name))
                else:
                    ham_list.append(get_text_from_email(email.message_from_file(f), file_name))
        except OSError:
            pass


print(len(ham_list))
df = pd.DataFrame(ham_list)
ham_file = os.path.join(os.getcwd(),'Dataset','lingspam_public','ham.csv')
df.to_csv(ham_file)

print(len(spam_list))
df = pd.DataFrame(spam_list)
spam_file = os.path.join(os.getcwd(),'Dataset','lingspam_public','spam.csv')
df.to_csv(spam_file)


