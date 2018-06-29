from __future__ import print_function
import os
import pandas as pd
import json
import sys

input_file = os.path.join(os.getcwd(),'Results', 'features', 'removed_filtered_all_features_v4.out')
output_file = os.path.join(os.getcwd(),'Results', 'features', 'unique_features_list.dmp')


df = pd.read_csv(input_file, encoding='latin-1')

features = list(set(df['feature']))
features = map(lambda x: x + '\n', features)
with open(output_file, 'w', encoding='latin-1') as f:
	f.writelines(features)