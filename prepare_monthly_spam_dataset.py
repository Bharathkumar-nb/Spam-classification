# imports:
import sys
import time
import datetime
import logging
import os
import numpy as np
import pandas as pd
import re
import random
import email
from nltk.corpus import stopwords

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

# Check command line arguments
if len(sys.argv) != 2:
    print('File Usage: python prepare_spam_dataset.py <src-folder>')
    print('Location of <src-folder>: Dataset/')
    sys.exit(0)

# Parse the email and extract right contents as text and returns dict - {'content': '...'}
def get_text_from_email(msg):
    parts = {}
    parts['content'] = ' '
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload()
            if payload != None:
                parts['content'] += payload
    
    if msg["Date"]:
        try:
            e_date = email.utils.parsedate_tz(msg['Date'])
            if e_date:
                date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(e_date))
                parts['content'] += ' weekday_' +  str(date.weekday()) + ' hour_of_day_' + str(date.hour)
        except Exception:
            pass
    if msg['Subject']:
        parts['content'] += ' ' + msg['Subject']
    return parts

# input spam folder
spam_folder = os.path.join(os.getcwd(),'Dataset', sys.argv[1])


for subdir, dirs, files in os.walk(spam_folder):
    if dirs != []:
        continue
    spam_list = []
    for filename in files:
        try:
            with open(os.path.join(subdir, filename),encoding='latin-1') as f:
                spam_list.append(get_text_from_email(email.message_from_file(f)))
        except OSError:
            pass
    df = pd.DataFrame(spam_list)
    # Output file path. Saves as .csv file
    spam_file = os.path.join(os.getcwd(),'Dataset', 'monthly_dataset_csv', sys.argv[1]+'_'+str(int(os.path.basename(os.path.normpath(subdir))))+'_spam.csv')
    df.to_csv(spam_file)


