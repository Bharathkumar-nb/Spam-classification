import os
import sys
import pandas as pd

if len(sys.argv)<2:
	print("File Usage: python synthetic_attack_prepare_data.py all_mails0_075.csv")
	sys.exit(1)

input_file = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', sys.argv[1])
target_folder = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', sys.argv[1].split('.')[0])

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

df = pd.read_csv(input_file, encoding='latin-1')

df = df.sample(frac=1)

no_of_rows = df.shape[0]

new_no_of_rows = no_of_rows//10

current_start_idx = 0

for i in range(10):
	target_file = os.path.join(target_folder, "df{}.csv".format(i))
	if i == 9:
		tmp_df = pd.DataFrame(df[current_start_idx:], columns=df.columns)
	else:
		tmp_df = pd.DataFrame(df[current_start_idx:current_start_idx+new_no_of_rows], columns=df.columns)
	current_start_idx = current_start_idx+new_no_of_rows
	print(i, tmp_df.shape)
	tmp_df.to_csv(target_file)