import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_who_do_new.pkl','rb') as f:
    probs_who_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_who_po_new.pkl','rb') as f:
    probs_who_po = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_what_do_new.pkl','rb') as f:
    probs_what_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_what_po_new.pkl','rb') as f:
    probs_what_po = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))

who_log_ratio = [probs_who_do[j] - probs_who_po[j] for j in range(len(probs_who_do))]
what_log_ratio = [probs_what_do[j] - probs_what_po[j] for j in range(len(probs_what_do))]
print(np.array(who_log_ratio))

print("t = " + str(calculate_t(np.array(who_log_ratio),np.array(what_log_ratio))))

fig, axis = plt.subplots()
probs = [who_log_ratio,what_log_ratio]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["who","what"])
axis.set_xlabel("Types of questions")
axis.set_ylabel("Log likelihood")
plt.gca().invert_yaxis()
plt.show()

