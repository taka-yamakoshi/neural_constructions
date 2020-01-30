import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_good_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_po_'+args[1]+'.pkl','rb') as f:
    probs_good_po = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_do_'+args[1]+'.pkl','rb') as f:
    probs_bad_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_bad_po_'+args[1]+'.pkl','rb') as f:
    probs_bad_po = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


good_log_ratio = [probs_good_do[j] - probs_good_po[j] for j in range(len(probs_good_do))]
bad_log_ratio = [probs_bad_do[j] - probs_bad_po[j] for j in range(len(probs_bad_do))]
print("t = " + str(calculate_t(np.array(good_log_ratio),np.array(bad_log_ratio))))

fig, axis = plt.subplots()
probs = [good_log_ratio,bad_log_ratio]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Alternating verbs","Non-alternating Verbs"])
axis.set_xlabel("Types of verbs")
axis.set_ylabel("Log likelihood ratio")
plt.gca().invert_yaxis()
plt.show()

