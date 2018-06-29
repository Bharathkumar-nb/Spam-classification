# imports:
import os
import sys
import pandas as pd

yearly_spam_folder = os.path.join(os.getcwd(),'Dataset', 'Train')
# skip_files = ['2012_spam1.csv', '2012_spam2.csv', '2013_spam1.csv', '2013_spam2.csv', '2006_spam1.csv',
#             '2006_spam2.csv', '2014_spam1.csv', '2014_spam2.csv', 'ham.csv', 'ham_latin.csv']
skip_files = ['ham.csv', 'ham_latin.csv']

if (len(sys.argv) < 3):
    print("File Usage: python allyears_prepare_spam_dataset.py [0/1] [30000/0.08]")
    sys.exit(1)

dfs = []

target_file = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', 'all_mails'+sys.argv[2].replace('.','_')+'.csv')

for subdir, dirs, files in os.walk(yearly_spam_folder):
    if dirs != []:
        continue
    for spam_file in files:
        if spam_file not in skip_files:
            print('file:{}'.format(spam_file))
            spam_df = pd.read_csv(os.path.join(subdir, spam_file), encoding='latin-1')
            spam_df.dropna(inplace=True)
            spam_df['label'] = 1
            if sys.argv[1] == '0':
                spam_df = spam_df.sample(min(int(sys.argv[2]),spam_df.shape[0]))
            else:
                spam_df = spam_df.sample(frac=float(sys.argv[2]))
                
            print(spam_df.shape)
            print("\n-------------------------\n")
            dfs.append(spam_df)

final_df = pd.concat(dfs)

final_df.to_csv(target_file)