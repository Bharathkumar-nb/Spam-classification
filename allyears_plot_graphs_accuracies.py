# imports
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, zero_one_loss, classification_report, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pickle

if len(sys.argv) < 3:
    print("File Usage: python allyears_plot_graphs_accuracies.py model_all_mails0_075.pkl vectorizer_all_mails0_075.pkl\
        \nModels are in Results/all_year_mails/")
    sys.exit(1)

yearly_spam_folder = os.path.join(os.getcwd(),'Dataset', 'Test')
model_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[1])
vectorizer_file = os.path.join(os.getcwd(),'Results', 'all_year_mails', sys.argv[2])

classifier = joblib.load(model_file)
with open(vectorizer_file, 'rb') as f:
    vocabulary = pickle.load(f)
vectorizer = CountVectorizer(vocabulary=vocabulary)


title = 'Accuracy of model trained on ' + sys.argv[1].split('.')[0]
x_label = []
y_label = []
for subdir, dirs, files in os.walk(yearly_spam_folder):
    if dirs != []:
        continue
    for f in files:
        filepath = os.path.join(subdir, f)
        df = pd.read_csv(filepath, encoding='latin-1')
        if 'ham' in f:
            df['label'] = 0
            accuracy = accuracy_score(classifier.predict(vectorizer.transform(df['content'])), df['label'])
            print("Accuracy for {}: {}".format(f, accuracy))
        else:
            df['label'] = 1
            predictions = classifier.predict(vectorizer.transform(df['content']))
            accuracy = accuracy_score(predictions, df['label'])
            year = int(f.split('_')[0])
            x_label.append(year)
            y_label.append(accuracy)
            print('Year: {}'.format(year))
            print('accuracy_score: {}'.format(accuracy_score(predictions, df['label'])))
            print('zero_one_loss: {}'.format(zero_one_loss(predictions, df['label'])))
            print('classification_report:\n{}'.format(classification_report(predictions, df['label'])))
            print('confusion_matrix:\n{}'.format(confusion_matrix(predictions, df['label'])))
            print()

sorted_x_label = sorted(x_label)
sorted_y_label = sorted(y_label, key=lambda x:x_label[y_label.index(x)])

plt.figure()
lw = 2
plt.plot(sorted_x_label, sorted_y_label, color='darkorange', lw=lw, label=title)
plt.xlabel('Test Datasets')
plt.ylabel('Accuracy')
plt.title(title)
# plt.show()
targetfile = sys.argv[1].split('.')[0] + '.png'
targetfilepath = os.path.join(os.getcwd(),'Results', 'all_year_mails', 'Accuracy',targetfile)
plt.savefig(targetfilepath)
plt.close()