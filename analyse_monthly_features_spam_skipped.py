from __future__ import print_function
import os
import pandas as pd
import json
import sys

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
for yr_,m_ in dates:
    filename = str(yr_)+'_'+str(m_)+'_spam_features.out'
    print(filename)
    tmp_df = pd.read_csv(os.path.join(features_folder, filename))

    for (yr, m) in dates:
        if int(m) not in [1,6]:
            continue 
        print(yr,m)
        filename = str(yr) + '_' + str(m) + '_spam.csv'
        #print(*list(df[df['file']==filename]['feature']), sep = "\n")
        top_features = list(df[df['file']==filename]['feature'])

        #print(f)
        #print(tmp_df[tmp_df['feature'].isin(top_features)]['feature'])
        #print('-------')
        #print(set(top_features) - set(tmp_df[tmp_df['feature'].isin(top_features)]['feature']))
        #non_existant_features = set(top_features) - set(tmp_df[tmp_df['feature'].isin(top_features)]['feature'])
        if str((yr,m)) not in dic:
            dic[str((yr,m))] = {}
        for feature in top_features:
            if feature not in dic[str((yr,m))]:
                dic[str((yr,m))][feature] = []
            #print(tmp_df[tmp_df['feature'] == feature])
            tmp = tmp_df[tmp_df['feature'] == feature]
            #print(tmp.shape[0])
            weight = 0 if (tmp.shape[0]==0) else tmp.iloc[0]['weight']
            dic[str((yr,m))][feature].append(weight)
            #break
        #eprint(dic)
        #break
print(dic)

json.dump(dic, open(os.path.join(os.getcwd(),'Results', 'features', 'graph_input_filtered_removed_skipped.json'),'w'))