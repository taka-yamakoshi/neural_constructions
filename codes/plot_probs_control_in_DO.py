import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_good_do_bert.pkl','rb') as f:
    probs_pron = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_control = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))

print("t = " + str(calculate_t(np.array(probs_pron),np.array(probs_control))))

fig, axis = plt.subplots()
probs = [probs_pron,probs_control]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Pronouns","Nouns with indefinite articles"])
axis.set_xlabel("Types of objects")
axis.set_ylabel("Log likelihood")
plt.gca().invert_yaxis()
plt.show()
