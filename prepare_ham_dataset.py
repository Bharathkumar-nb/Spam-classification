# imports:
import time;
import datetime
import logging
import os
import sys
import numpy as np
import pandas as pd
import re
import random
import email
from nltk.corpus import stopwords

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def get_text_from_email(msg):
    parts = {}
    parts['content'] = ''
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload()
            if payload != None:
                parts['content'] += payload
    
    if msg["Date"]:
        e_date = email.utils.parsedate_tz(msg['Date'])
        if e_date:
            date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(e_date))
            parts['content'] += ' weekday_' +  str(date.weekday()) + ' hour_of_day_' + str(date.hour)
    if msg['Subject']:
        # parts['subject'] = msg['Subject']
        parts['content'] += ' ' + msg['Subject']
    return parts

ham_folder = os.path.join(os.getcwd(),'Dataset', 'Ham')

ham_list = []

for subdir, dirs, files in os.walk(ham_folder):
    for file_n in files:
        with open(os.path.join(subdir, file_n), encoding='latin-1') as f:
            ham_list.append(get_text_from_email(email.message_from_file(f)))

print(len(ham_list))
df = pd.DataFrame(ham_list)
ham_file = os.path.join(os.getcwd(),'Dataset','csv_files','ham.csv')
df.to_csv(ham_file)
ham_file = os.path.join(os.getcwd(),'Dataset','csv_files','ham_latin.csv')
df.to_csv(ham_file,encoding='latin-1')
