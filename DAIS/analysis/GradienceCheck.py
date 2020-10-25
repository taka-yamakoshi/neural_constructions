import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import sys
sys.path.append('../..')
args = sys.argv

raw_data  = pd.read_csv('../data/experiment_output/data_cleaned.csv')
behav_data_pronoun = np.array([raw_data.query(f'verb_id == {id+1} and recipient_id == "pronoun"')['DOpreference'].to_numpy().astype(float) for id in range(200)])
ave_behav = np.array([raw_data.query(f'verb_id == {id+1} and recipient_id == "pronoun"')['DOpreference'].mean() for id in range(200)])
sorted_verb_id = np.argsort(ave_behav)

verb_data = pd.read_csv('../data/experiment_input/verblists.csv')
verb_list = [list(verb_data.query(f'verb_id == {id+1}')['verb'])[0] for id in range(200)]

color_list = sns.color_palette('Set2')
fig = plt.figure(figsize=(15.5, 20),dpi=150)
ymax_list = [40,30,30,20,20,20,20,20,20,20,
             20,20,20,20,20,20,20,25,20,45]
for pos,id in enumerate(sorted_verb_id):
    ax1 = fig.add_subplot(17,12,pos+1)
    ax2 = ax1.twinx()
    kde_model = gaussian_kde(behav_data_pronoun[id],bw_method='scott')
    x_grid = np.linspace(0,101,num=500)
    if id < 100:
        ax1.hist(behav_data_pronoun[id],color=color_list[1],alpha=0.5,bins=np.arange(0,101,10))
        ax2.plot(x_grid,kde_model(x_grid), color=color_list[1],linewidth=2)
    else:
        ax1.hist(behav_data_pronoun[id],color=color_list[0],alpha=0.5,bins=np.arange(0,101,10))
        ax2.plot(x_grid,kde_model(x_grid), color=color_list[0],linewidth=2)

    ax1.set_xticks([0,50,100])
    if pos < 188:
        ax1.set_xticklabels([])
    else:
        ax1.set_xticklabels([0,50,100],fontsize=8,rotation=90)
    ymax = ymax_list[pos//12]
    ax1.set_ylim(0,ymax)
    if pos % 12==0:
        ax1.set_yticks([0,int(ymax/2),ymax])
        ax1.set_yticklabels([0,int(ymax/2),ymax])
    else:
        ax1.set_yticks([])
    ax2.set_ylim([0,0.05])
    ax2.set_yticks([])
    plt.title(verb_list[id],fontsize=13)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
fig.subplots_adjust(hspace=0.75,bottom=0.05,top=0.95,left=0.05,right=0.95)
fig.savefig('../raw_dist.pdf')
