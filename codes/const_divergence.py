import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import sys


sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/probs_from_const_good_do_new.pkl','rb') as f:
    probs_new = pickle.load(f)
with open(PATH + 'datafile/probs_from_const_good_do_control_1.pkl','rb') as f:
    probs_control_1 = pickle.load(f)
with open(PATH + 'datafile/probs_from_const_good_do_control_2.pkl','rb') as f:
    probs_control_2 = pickle.load(f)


def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))

def kl_div(x,y):
    return np.sum(np.array([x[i] * np.log(x[i]/y[i]) for i in range(x.size)]))
order = np.array([list(np.argsort(probs)[::-1][:10]) for probs in probs_new])

div_from_1 = np.array([kl_div(np.exp(probs_new[i][order[i]]),np.exp(probs_control_1[i][order[i]])) for i in range(probs_new.shape[0])])
div_from_2 = np.array([kl_div(np.exp(probs_new[i][order[i]]),np.exp(probs_control_2[i][order[i]])) for i in range(probs_new.shape[0])])
print(div_from_1-div_from_2)
print("t = " + str(calculate_t(div_from_1,div_from_2)))

fig, axis = plt.subplots()
probs = [div_from_1,div_from_2]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Divergence from control 1","Divergence from control 2"])
axis.set_xlabel("Controls")
axis.set_ylabel("KL Divergence")
plt.show()

