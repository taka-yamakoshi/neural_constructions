from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import random
from multiprocessing import Pool
import sys
import seaborn as sns
sns.set()

sys.path.append('..')
args = sys.argv

with open('../data/PrunedGeneratedSentsSWBDNewResults.csv','r') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
head = file[0]
text = file[1:]

model_list = ['ngram','lstm','bert','gpt2','gpt2-large']
scores = np.empty(len(model_list))
fig = plt.figure(figsize=(10,2))
for i,model_name in enumerate(model_list):
    prob_ratio = np.array([[float(row[head.index(f'{model_name}_ratio')]), int(row[head.index('realized_construction')]=='DO')] for row in text])
    X = prob_ratio[:,0].reshape((prob_ratio.shape[0],1))
    y = prob_ratio[:,1]
    model = LogisticRegression(penalty='none',solver='lbfgs')
    model.fit(X,y)
    plt.subplot(1,len(model_list),i+1)
    plt.scatter(model.coef_[0][0]*prob_ratio[:,0]+model.intercept_[0],prob_ratio[:,1])
    plt.yticks([])
    if i==0:
        plt.yticks([0,1],['PD','DO'])
    scores[i] = model.score(X,y)
    plt.title(model_name)
    plt.text(0,0.75,f'{scores[i]:.4f}')
    fig.subplots_adjust(bottom=0.2)

plt.show()
print(scores)
plt.bar(np.arange(len(model_list)),scores)
plt.xticks(np.arange(len(model_list)),model_list)
plt.show()

pred_list = model.predict(X)

with open('../data/PrunedGeneratedSentsSWBDNewResults.csv','r') as f:
    reader = csv.reader(f)
    file  = [row for row in reader]
    head = file[0]
    text = file[1:]

assert len(pred_list) == len(text)

with open('../data/PrunedGeneratedSentsSWBDNewResults.csv','w') as f:
    writer = csv.writer(f)
    if 'prediction_gpt2-large' in head:
        writer.writerow(head)
        for pred,row in zip(pred_list,text):
            row[head.index('prediction_gpt2-large')] = ['PD','DO'][int(pred)]
            writer.writerow(row)
    else:
        head.append('prediction_gpt2-large')
        writer.writerow(head)
        for pred,row in zip(pred_list,text):
            row.append(['PD','DO'][int(pred)])
            writer.writerow(row)
