from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import random
from multiprocessing import Pool
import sys

sys.path.append('..')
args = sys.argv

#Main figure
score_data = [np.zeros((3,10,4)),np.zeros((3,10,1)),np.zeros((3,10,13)),np.zeros((3,10,37))]
models = ['lstm','lstm-large','gpt2','gpt2-large']
positions = ['verb','first_obj','eos']
for i, model in enumerate(models):
    for j, position in enumerate(positions):
        for k in range(10):
            with open('../data/test_performance_'+model+'_'+position+'_'+str(k)+'.pkl','rb') as f:
                score_data[i][j][k] = pickle.load(f)

score_ave = [np.average(model_score,axis=1) for model_score in score_data]
score_ave_max = [np.max(model_score_ave,axis=1) for model_score_ave in score_ave]
score_ave_arg_max = [np.argmax(model_score_ave,axis=1) for model_score_ave in score_ave]
error = np.zeros((4,3,10))
for i in range(4):
    for j in range(3):
        error[i][j] = score_data[i][j][:,score_ave_arg_max[i][j]]

fig, axis = plt.subplots()
plot_list_LSTM = axis.bar([0,5,10],score_ave_max[0],yerr=np.std(error[0],axis=1)/10,color='lightseagreen',label='LSTM')
plot_list_LSTM = axis.bar([1,6,11],score_ave_max[1],yerr=np.std(error[0],axis=1)/10,color='teal',label='LSTM-large')
plot_list_LSTM = axis.bar([2,7,12],score_ave_max[2],yerr=np.std(error[0],axis=1)/10,color='salmon',label='GPT2')
plot_list_LSTM = axis.bar([3,8,13],score_ave_max[3],yerr=np.std(error[0],axis=1)/10,color='crimson',label='GPT2-large')
plt.legend(bbox_to_anchor=(0.875, 1.03), loc='upper right', borderaxespad=1,fontsize=8)
plt.xticks([1.5,6.5,11.5],["End of the verb","End of the first object","End of the sentence "])
axis.set_xlabel("Timing")
axis.set_ylabel("Explained Variance")
plt.show()
