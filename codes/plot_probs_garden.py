import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/log_probs_bad_do_bert.pkl','rb') as f:
    probs_bad = pickle.load(f)
with open(PATH + 'datafile/log_probs_bad_do_garden.pkl','rb') as f:
    probs_garden = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))
print("t = " + str(calculate_t(np.array(probs_bad),np.array(probs_garden))))

fig, axis = plt.subplots()
probs = [probs_bad,probs_garden]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Ungrammatical","Garden path"])
axis.set_xlabel("Types of constructions")
axis.set_ylabel("Log probability")
plt.gca().invert_yaxis()
plt.show()

