import pickle
import numpy as np
import csv
import logging
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

#Load sentences
with open(PATH+'textfile/generated_pairs_probs.csv') as f:
    reader = csv.reader(f)
    prob_file = [row for row in reader]
    prob_head = prob_file[0]
    corpus = prob_file[1:]

with open(PATH+'data/data_cleaned.csv') as f:
    reader = csv.reader(f)
    behav_file = [row for row in reader]
    behav_head = behav_file[0]
    judgement = behav_file[1:]

verb_id = behav_head.index('verb_id')
theme_id = behav_head.index('theme_id')
recipient_id = behav_head.index('recipient_id')
preference_id = behav_head.index('DOpreference')

recipient_type = ['pronoun','shortDefinite','longDefinite','shortIndefinite','longIndefinite']
behav_data = {}
for v in range(200):
    behav_data[str(v+1)] = {}
    for t in range(5):
        behav_data[str(v+1)][str(t+1)] = {}
        for r in recipient_type:
            behav_data[str(v+1)][str(t+1)][r] = []

for row in judgement:
    behav_data[row[verb_id]][row[theme_id]][row[recipient_id]].append(int(row[preference_id]))

with open(PATH+'datafile/behav_data.pkl','wb') as f:
    pickle.dump(behav_data,f)

with open(PATH+'textfile/generated_pairs_behav.csv','w') as f:
    writer = csv.writer(f)
    prob_head.append('BehavDOpreference')
    writer.writerow(prob_head)
    for row in corpus:
        verb = row[prob_head.index('verb_id')]
        theme = row[prob_head.index('theme_id')]
        recipient = row[prob_head.index('recipient_id')]
        row.append(np.average(np.array(behav_data[verb][theme][recipient])))
        writer.writerow(row)
