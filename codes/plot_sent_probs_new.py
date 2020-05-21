import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_prob_ratio_'+args[1]+'.pkl','rb') as f:
    ratio_1 = pickle.load(f)
with open(PATH + 'datafile/sent_prob_ratio_'+args[2]+'.pkl','rb') as f:
    ratio_2 = pickle.load(f)

with open(PATH + 'datafile/lstm_log_ratio_shortDefinite.pkl','rb') as f:
    ratio = pickle.load(f)

fig, axis = plt.subplots()
axis.bar(np.arange(2),np.array([np.average(ratio[i]) for i in range(2)]),yerr =np.array([np.std(ratio[i])/np.sqrt(len(ratio[i])) for i in range(2)]))
plt.show()
exit()



def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


fig, axis = plt.subplots()
probs = [ratio_1,ratio_2]

log_ratio_list = axis.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],[args[1],args[2]])
axis.set_xlabel("Types of recipients",fontsize = 15)
axis.set_ylabel("Log likelihood ratio",fontsize = 15)
axis.text(-0.05, 0, 'Double Object',
        transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
axis.text(-0.05, 0.75, 'Preposition Dative',
          transform=axis.transAxes,fontsize=15,rotation = 90,color = 'green')
plt.gca().invert_yaxis()
plt.show()

