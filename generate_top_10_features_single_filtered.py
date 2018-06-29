import os
import pandas as pd
import csv


features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full2')
common_folder = os.path.join(os.getcwd(),'Results', 'features')

flag = True
with open(os.path.join(common_folder, 'removed_filtered_all_features_v4.out'), 'w', encoding='latin-1') as g:
    with open(os.path.join(common_folder, 'removed_features_list.dmp'), 'r', encoding='latin-1') as h:
        removed_features = [line.strip() for line in h.readlines()]
        #print(removed_features)
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
                        i = -1
                        j = 0
                        while j<10:
                            line = lines[i].strip().split(",")
                            #print(line[2])
                            if line[2] not in removed_features:
                                c.writerow(line)
                                j += 1
                            i -= 1
                except OSError:
                    pass