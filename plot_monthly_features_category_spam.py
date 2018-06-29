from __future__ import print_function
import os
import pandas as pd
import json
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

input_file = os.path.join(os.getcwd(),'Results', 'features', 'removed_filtered_all_features_v4.out')
features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full2')

df = pd.read_csv(input_file, encoding='latin-1')
#print(set(df['file']))
files = set(df['file'])

dates = []

for filename in files:
    year, month, _ = filename.split('_')
    dates.append((int(year), int(month)))

#print(sorted(dates))

dates = sorted(dates)



dic = {}

graph_input_file = os.path.join(os.getcwd(),'Results', 'features', 'unique_features_list.dmp')
with open(graph_input_file, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    dic = {line[0]:line[1] for line in lines}

for (yr, m) in dates:
    print(yr,m)
    filename = str(yr) + '_' + str(m) + '_spam.csv'
    graph_output_file = os.path.join(os.getcwd(),'Results', 'features', 'wordDistribution', str(yr) + '_' + str(m) + '_spam.png')
    top_features = list(df[df['file']==filename]['feature'])
    categories = []
    for feature in top_features:
        if feature not in dic:
            categories.append('Unknown')
        else:
            categories.append(dic[feature])
    x = []
    y = []
    for category in set(categories):
        x.append(category)
        y.append(categories.count(category))

    #print(categories)

    plt.pie(y, labels=x, autopct='%1.1f%%', shadow=True)
    plt.savefig(graph_output_file)
    plt.clf()
