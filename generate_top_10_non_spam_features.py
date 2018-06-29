import os
import pandas as pd
import csv

features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full2')
output_folder = os.path.join(os.getcwd(),'Results', 'features')
flag = True
with open(os.path.join(output_folder, 'filtered_all_features_nonspam.out'), 'w', encoding='latin-1') as g:
    c = csv.writer(g)
    for subdir, dirs, files in os.walk(features_folder):
        for filename in files:
            print(filename)
            try:
                with open(os.path.join(subdir, filename), 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    if flag:
                        c.writerow(lines[0].strip().split(","))
                        flag = False
                    for i in range(1,11):
                        c.writerow(lines[i].strip().split(","))
            except OSError:
                pass