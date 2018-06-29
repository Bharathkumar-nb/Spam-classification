import os
import sys
import json
from ast import literal_eval
from cycler import cycler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
import pandas as pd

if len(sys.argv) < 2:
    print("File Usage: python allyears_plt_graph_features.py model_all_mails0_075.features\n\
        Features are in Results/all_year_mails/Features/")
    sys.exit(1)

input_features_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'Features', sys.argv[1])
features_folder = os.path.join(os.getcwd(),'Results', 'features', 'monthly_features_full')
graph_output_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'Graphs', sys.argv[1].split('.')[0]+'_ham.png')

df = pd.read_csv(input_features_file, encoding='latin-1')
df.sort_values(by=['weight', 'feature'], ascending=[False, True], inplace=True)

#df = pd.DataFrame(df[:10], columns=['weight', 'feature'])
top_features = list(df['feature'][-10:])
top_features_weights = list(df['weight'][-10:])

del df

x_label = []
y_label = {}
for feature in top_features:
    y_label[feature] = []
for subdir, dirs, files in os.walk(features_folder):
    for filename in files:
        yr, m, _, _ = filename.split('_')
        if m != '6':
            continue
        print(yr)
        yr = int(yr)
        tmp_df = pd.read_csv(os.path.join(features_folder, filename))

        x_label.append(yr)
        for feature in top_features:
            tmp = tmp_df[tmp_df['feature'] == feature]
            weight = 0 if (tmp.shape[0]==0) else tmp.iloc[0]['weight']
            y_label[feature].append(weight)

for feature in top_features:
    y_label[feature] = sorted(y_label[feature], key=lambda x:x_label[y_label[feature].index(x)])

x_label.sort()

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'violet', 'tan', 'teal']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'])))

fig, ax = plt.subplots()
for feature, weights in y_label.items():
    ax.plot(x_label, weights, label="{} ({})".format(feature, round(top_features_weights[top_features.index(feature)], 4)), linewidth=1)
ax.set_xlabel('Years (Every June)')
ax.set_ylabel('Weights')
plt.title("Comparison of Top 10 Ham Features")
fig.legend()
fig.savefig(graph_output_file)