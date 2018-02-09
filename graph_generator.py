# imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
for i in range(1, len(sys.argv)):
    filepath = os.path.join(os.getcwd(),'Results',sys.argv[i])
    with open(filepath) as f:
        year = sys.argv[i].split('_')[0]
        title = 'Accuracy of model trained on ' + year
        x_label = []
        y_label = []
        for line in f.readlines():
            if '_spam_on_' in line:
                x = int(line.split('_spam_on_')[1][:4])
                x_label.append(x)
            elif '_spam1_on_' in line:
                x = int(line.split('_spam_on_')[1][:4])
                x_label.append(x)
            elif '_spam2_on_' in line:
                x = int(line.split('_spam_on_')[1][:4])
                x_label.append(x)
            if 'accuracy_score' in line and len(y_label)<len(x_label):
                y = round(float(line.split(' ')[1]),2)
                y_label.append(y)
        if len(x_label) != len(y_label):
            print(len(x_label), len(y_label))
            sys.exit(1)
        plt.figure()
        lw = 2
        plt.plot(x_label, y_label, color='darkorange', lw=lw, label=title)
        plt.xlabel('Test Datasets')
        plt.ylabel('Accuracy')
        plt.title(title)
        # plt.show()
        targetfile = '{}_model.png'.format(year)
        targetfilepath = os.path.join(os.getcwd(),'Results','Accuracy',targetfile)
        plt.savefig(targetfilepath)
        plt.close()
