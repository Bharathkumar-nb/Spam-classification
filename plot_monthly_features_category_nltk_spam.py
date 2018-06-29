from __future__ import print_function
import os
import pandas as pd
import json
import sys
import csv
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nltk.download('wordnet')
from nltk.corpus import wordnet
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full2')

yrs = [x for x in range(2000, 2017)]
months = [x for x in range(1,13)]
dates = []

for yr in yrs:
    for m in months:
        dates.append((yr, m))

dates = sorted(dates)

dic = {}

graph_input_file = os.path.join(os.getcwd(),'Results', 'features', 'unique_features_list.dmp')
with open(graph_input_file, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    dic = {line[0]:line[1] for line in lines}


for (yr, m) in dates:
    filename = str(yr) + '_' + str(m) + '_spam_features.out'
    print(filename)
    graph_input_file = os.path.join(features_folder, filename)
    f = open(graph_input_file, 'r', encoding='latin-1')
    lines = f.readlines()
    data = StringIO('\n'.join([lines[0]] + lines[-101:]))

    df = pd.read_csv(data, sep=',')
    graph_output_file = os.path.join(os.getcwd(),'Results', 'features', 'wordDistribution', str(yr) + '_' + str(m) + '_spam.png')
    i = 1
    categories = {}
    for index,row in df.iterrows():
        if i > 100:
            break
        if row['feature'] not in dic:
            if row['feature'] in wordnet.words():
                count, total = categories.get('dictionary_word', (0,0.0))
                categories['dictionary_word'] = (count+1, total+row['weight'])
            else:
                count, total = categories.get('non_dictionary_word', (0,0.0))
                categories['non_dictionary_word'] = (count+1, total+row['weight'])
        else:
            count, total = categories.get(dic[row['feature']], (0,0.0))
            categories[dic[row['feature']]] = (count+1, total+row['weight'])
        i += 1
    x = []
    y = []
    for category, (count,total) in categories.items():
        avg = 0
        if count != 0:
            avg = total/count
        x.append("{}\navg weight({})".format(category, avg))
        y.append(count)
    plt.figure(figsize=(10,10))
    plt.title('Top ' + str(sum(y)) + ' features composition')
    plt.pie(y, labels=x, autopct='%1.1f%%', shadow=True)
    plt.savefig(graph_output_file)
    plt.clf()
    #break
