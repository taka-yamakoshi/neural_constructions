import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_passive_good.pkl','rb') as f:
    probs_good = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_passive_bad.pkl','rb') as f:
    probs_bad = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))

print("t = " + str(calculate_t(np.array(probs_good),np.array(probs_bad))))

fig, axis = plt.subplots()
probs = [probs_good,probs_bad]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Good","Bad"])
axis.set_xlabel("Types of Constructions")
axis.set_ylabel("Log Probability")
plt.gca().invert_yaxis()
axis.set_ylim(-25,-35)
plt.show()

