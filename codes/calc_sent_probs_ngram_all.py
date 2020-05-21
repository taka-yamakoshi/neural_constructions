import pickle
import numpy as np
import csv
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

#Load the model
import kenlm
model = kenlm.Model('../../kenlm/lm/text.arpa')

#Load sentences
with open(PATH+'textfile/generated_pairs_behav.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    corpus = file[1:]
sent_list = [[pair[head.index('DOsentence')],pair[head.index('PDsentence')]] for pair in corpus]


DO_prob = np.array([model.score(sentences[0], bos = True, eos = True) for sentences in sent_list])
PD_prob = np.array([model.score(sentences[1], bos = True, eos = True) for sentences in sent_list])
ratio = DO_prob - PD_prob
with open(PATH+'datafile/ngram_DO_test.pkl','wb') as f:
    pickle.dump(DO_prob,f)
with open(PATH+'datafile/ngram_PD_test.pkl','wb') as f:
    pickle.dump(PD_prob,f)
with open(PATH+'datafile/ngram_log_ratio_test.pkl','wb') as f:
    pickle.dump(ratio,f)

with open(PATH+'textfile/generated_pairs_ngram_test.csv','w') as f:
    writer = csv.writer(f)
    head.extend(['ngram_ratio'])
    writer.writerow(head)
    for i,row in enumerate(corpus):
        row.extend([ratio[i]])
        writer.writerow(row)
