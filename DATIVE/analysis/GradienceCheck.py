import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import n_colors
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr,beta,kurtosis
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
import sys
sys.path.append('../..')
args = sys.argv

raw_data  = pd.read_csv('../data/experiment_output/data_cleaned.csv')
bahav_data = np.array([raw_data.query(f'verb_id == {id+1} and recipient_id == "pronoun"')['DOpreference'].to_numpy().astype(float) for id in range(200)])

verb_data = pd.read_csv('../data/experiment_input/verblists.csv')
verb_list = [list(verb_data.query(f'verb_id == {id+1}')['verb'])[0] for id in range(200)]

#Fit beta distribution and calculate kurtosis
beta_params = np.empty((200,2))
kurtosis_list = np.empty(200)
for verb_id,verb_behav in enumerate(bahav_data):
    verb_behav[verb_behav==0] = 1
    verb_behav[verb_behav==100] = 99
    verb_behav = verb_behav/100
    a,b,loc,scale = beta.fit(verb_behav,floc=0,fscale=1)
    beta_params[verb_id] = [a,b]
    print(a,b)
    kurtosis_list[verb_id] = kurtosis(verb_behav)
print(f'Verbs with both parameters>1: {np.sum([row[0]>1 and row[1]>1 for row in beta_params])}')
print(f'Verbs with kurtosis>0: {np.sum(kurtosis_list>0)}')

#Violin plot
color_list = sns.color_palette('Set2')
fig = plt.figure(figsize=(10,10),dpi=150)
for verb_id,verb_behav in enumerate(bahav_data[:100]):
    plt.subplot(10,10,verb_id+1)
    if beta_params[verb_id][0]>1 and beta_params[verb_id][1]>1:
        sns.violinplot(verb_behav,color=color_list[0])
    elif beta_params[verb_id][0]<1 and beta_params[verb_id][1]<1:
        sns.violinplot(verb_behav,color=color_list[1])
    else:
        sns.violinplot(verb_behav,color=color_list[2])
    plt.title(verb_list[verb_id])
    if verb_id < 90:
        plt.xticks([0,50,100],[])
    else:
        plt.xticks([0,50,100],[0,50,100],fontsize=8,rotation=90)
fig.subplots_adjust(hspace=1,wspace=1)
fig.savefig('../non-alternating.png')

fig = plt.figure(figsize=(10,10),dpi=150)
for verb_id,verb_behav in enumerate(bahav_data[100:]):
    plt.subplot(10,10,verb_id+1)
    if beta_params[verb_id+100][0]>1 and beta_params[verb_id+100][1]>1:
        sns.violinplot(verb_behav,color=color_list[0])
    elif beta_params[verb_id+100][0]<1 and beta_params[verb_id+100][1]<1:
        sns.violinplot(verb_behav,color=color_list[1])
    else:
        sns.violinplot(verb_behav,color=color_list[2])
    plt.title(verb_list[verb_id+100])
    if verb_id < 90:
        plt.xticks([0,50,100],[])
    else:
        plt.xticks([0,50,100],[0,50,100],fontsize=8,rotation=90)
fig.subplots_adjust(hspace=1,wspace=1)
fig.savefig('../alternating.png')
