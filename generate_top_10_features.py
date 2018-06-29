import os
import pandas as pd
import csv

features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full')
output_folder = os.path.join(os.getcwd(),'Results', 'features', 'top_10')

for subdir, dirs, files in os.walk(features_folder):
    for filename in files:
        print(filename)
        try:
            with open(os.path.join(subdir, filename), 'r', encoding='latin-1') as f, open(os.path.join(output_folder, filename), 'w', encoding='latin-1') as g:
                lines = f.readlines()
                c = csv.writer(g)
                c.writerow(lines[0].strip().split(","))
                for i in range(-1,-11,-1):
                    c.writerow(lines[i].strip().split(","))
        except OSError:
            pass