# imports
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

dir_path = os.path.join(os.getcwd(),'Results','features')

df = []

for filename in os.listdir(dir_path):
    filepath = os.path.join(dir_path, filename)
    yr = filename.split('_')[1][:4]
    with open(filepath, encoding='latin-1') as f:
        lines = f.readlines()
        if len(lines)<2:
            continue
        for line in lines[1:]:
            try:
                line = line.split()
                wt = line[0]
                word = ' '.join(line[1:])
            except:
                print(line)
                continue
            row = {}
            row['weights'] = round(float(wt), 3)
            row['words'] = word
            row['year'] = yr
            df.append(row)


df = pd.DataFrame(df)
print(df)
print('--------------------')
print(df.groupby(['words']).size())
