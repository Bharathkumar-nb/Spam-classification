import os
import json
from ast import literal_eval
from cycler import cycler
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm 

graph_input_file = os.path.join(os.getcwd(),'Results', 'features', 'unique_features_list.dmp')
graph_output_file = os.path.join(os.getcwd(),'Results', 'features', 'wordDistribution', 'pie.png')

with open(graph_input_file, 'r') as csvfile:
	df = csv.reader(csvfile, delimiter=',')
	categories = []
	for row in df:
		categories.append(row[1])
	x = []
	y = []
	for category in set(categories):
		x.append(category)
		y.append(categories.count(category))

	plt.pie(y, labels=x, autopct='%1.1f%%', shadow=True)
	plt.savefig(graph_output_file)
	plt.clf()