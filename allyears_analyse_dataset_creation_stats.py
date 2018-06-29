# imports:
import os
import sys
import numpy as np

stat_file = os.path.join(os.getcwd(),'Dataset', 'all_year_mails_csv', 'stats.txt')

with open(stat_file) as f:
	no_of_mails_list = []
	yrs = []
	lines = f.readlines()
	for i in range(0, len(lines), 5):
		yr = lines[i].split(':')[1].split('_')[0]
		yrs.append(yr)
		after_shape = eval(lines[i+1])
		no_of_mails_list.append(after_shape[0])

	print(sorted(no_of_mails_list))
	min_idx = np.argmin(no_of_mails_list)
	print("Min year {}: {}".format(yrs[min_idx], no_of_mails_list[min_idx]))
	max_idx = np.argmax(no_of_mails_list)
	print("Max year {}: {}".format(yrs[max_idx], no_of_mails_list[max_idx]))
	print("Average: {}".format(sum(no_of_mails_list)/len(no_of_mails_list)))
	print("Total no of mails generated: {}".format(sum(no_of_mails_list)))