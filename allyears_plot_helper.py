import os
import sys
#from matplotlib.pyplot import cm
import pandas as pd

if len(sys.argv) < 2:
    print("File Usage: python allyears_plt_graph_features.py model_all_mails0_075.features\n\
        Features are in Results/all_year_mails/Features/")
    sys.exit(1)

input_features_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'Features', sys.argv[1])

df = pd.read_csv(input_features_file, encoding='latin-1')
df.sort_values(by=['weight', 'feature'], ascending=[False, True], inplace=True)

print('Spam')
top_features = list(df['feature'][:30])
top_features_weights = list(df['weight'][:30])

for i in range(len(top_features)):
	print(top_features[i], top_features_weights[i])

print("\nHam")
ham_top_features = list(df['feature'][-30:])
ham_top_features_weights = list(df['weight'][-30:])

for i in range(len(ham_top_features)-1, -1, -1):
	print(ham_top_features[i], ham_top_features_weights[i])