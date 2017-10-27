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

if len(sys.argv) != 3:
    print('File Usage: python prepare_spam_dataset.py <src-folder> <dst-file>')
    print('Location of <src-folder>|<dst-file> : Dataset/')
    sys.exit(0)

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


spam_folder = os.path.join(os.getcwd(),'Dataset', sys.argv[1])

spam_list = []

for subdir, dirs, files in os.walk(spam_folder):
    for filename in files:
        try:
            with open(os.path.join(subdir, filename),encoding='latin-1') as f:
                spam_list.append(get_text_from_email(email.message_from_file(f), filename))
        except OSError:
            pass

df = pd.DataFrame(spam_list)
spam_file = os.path.join(os.getcwd(),'Dataset', sys.argv[2])
df.to_csv(spam_file)


