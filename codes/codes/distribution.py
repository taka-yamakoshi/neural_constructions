import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
PATH = 'Path to the folder'

with open(PATH + 'datafile/log_probs_good_do_'+args[1]+'.pkl','rb') as f:
    probs_good_do = pickle.load(f)
with open(PATH + 'datafile/log_probs_good_po_'+args[1]+'.pkl','rb') as f:
    probs_good_po = pickle.load(f)
with open(PATH + 'datafile/log_probs_bad_do_'+args[1]+'.pkl','rb') as f:
    probs_bad_do = pickle.load(f)
with open(PATH + 'datafile/log_probs_bad_po_'+args[1]+'.pkl','rb') as f:
    probs_bad_po = pickle.load(f)


with open(PATH + 'textfile/good_do_'+args[1]+'.txt') as f:
    good_text = f.read().split('\n')[:-1]
with open(PATH + 'textfile/bad_do_'+args[1]+'.txt') as f:
    bad_text = f.read().split('\n')[:-1]

good_verb = [sentence.split(" ")[2] for sentence in good_text]
bad_verb = [sentence.split(" ")[2] for sentence in bad_text]
new_good_verb = [good_verb[2*i] for i in range(int(len(good_verb)/2))]
new_bad_verb = [bad_verb[2*i] for i in range(int(len(bad_verb)/2))]

new_good_do = [(probs_good_do[2*i]+probs_good_do[2*i+1])/2 for i in range(len(new_good_verb))]
new_good_po = [(probs_good_po[2*i]+probs_good_po[2*i+1])/2 for i in range(len(new_good_verb))]
new_bad_do = [(probs_bad_do[2*i]+probs_bad_do[2*i+1])/2 for i in range(len(new_bad_verb))]
new_bad_po = [(probs_bad_po[2*i]+probs_bad_po[2*i+1])/2 for i in range(len(new_bad_verb))]

new_good_ratio = [new_good_do[j] - new_good_po[j] for j in range(len(new_good_do))]
new_bad_ratio = [new_bad_do[j] - new_bad_po[j] for j in range(len(new_bad_do))]

order_good = np.argsort(np.array(new_good_ratio))[::-1]
order_bad = np.argsort(np.array(new_bad_ratio))[::-1]
with open(PATH + 'textfile/distribution_'+args[1]+'.txt','w') as f:
    f.write("Alternating verbs")
    f.writelines('\n')
    f.write("verb")
    f.writelines('\t')
    f.write("DO")
    f.writelines('\t')
    f.write("PO")
    f.writelines('\t')
    f.write("ratio")
    f.writelines('\n')
    for id in order_good:
        f.write(str(new_good_verb[id]))
        f.writelines('\t')
        f.write(str(new_good_do[id]))
        f.writelines('\t')
        f.write(str(new_good_po[id]))
        f.writelines('\t')
        f.write(str(new_good_ratio[id]))
        f.writelines('\n')
    f.writelines('\n')
    f.writelines('\n')
    f.writelines('\n')
    f.write("Non-alternating verbs")
    f.writelines('\n')
    f.write("verb")
    f.writelines('\t')
    f.write("DO")
    f.writelines('\t')
    f.write("PO")
    f.writelines('\t')
    f.write("ratio")
    f.writelines('\n')
    for id in order_bad:
        f.write(str(new_bad_verb[id]))
        f.writelines('\t')
        f.write(str(new_bad_do[id]))
        f.writelines('\t')
        f.write(str(new_bad_po[id]))
        f.writelines('\t')
        f.write(str(new_bad_ratio[id]))
        f.writelines('\n')
