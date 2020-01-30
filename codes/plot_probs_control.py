import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/sent_log_probs_good_do_bert.pkl','rb') as f:
    probs_pron_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_po_bert.pkl','rb') as f:
    probs_pron_po = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_control_do = pickle.load(f)
with open(PATH + 'datafile/sent_log_probs_good_po_'+args[1]+'.pkl','rb') as f:
    probs_control_po = pickle.load(f)

def calculate_t(x,y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    x_var = np.var(x)
    y_var = np.var(y)
    pooled_var = (x_var*(x.size-1)+y_var*(y.size-1))/(x.size+y.size-2)
    return (x_ave-y_ave)/(np.sqrt(pooled_var*(1/x.size+ 1/y.size)))


pron_log_ratio = [probs_pron_do[j] - probs_pron_po[j] for j in range(len(probs_pron_do))]
control_log_ratio = [probs_control_do[j] - probs_control_po[j] for j in range(len(probs_control_do))]
print("t = " + str(calculate_t(np.array(pron_log_ratio),np.array(control_log_ratio))))

fig, axis = plt.subplots()
probs = [pron_log_ratio,control_log_ratio]
plt.bar(np.arange(2),np.array([np.average(probs[i]) for i in range(2)]),yerr =np.array([np.std(probs[i])/np.sqrt(len(probs[i])) for i in range(2)]))
plt.xticks([0,1],["Pronouns","Control"])
axis.set_xlabel("Types of objects")
axis.set_ylabel("Log likelihood ratio")
plt.gca().invert_yaxis()
plt.show()

