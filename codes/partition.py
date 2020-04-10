import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

#Load sentences
with open(PATH+'textfile/generated_pairs.csv') as f:
    reader = csv.reader(f)
    corpus = [row for row in reader]
    head = corpus[0]
    body = corpus[1:]

order = [(1+5+25+2500)*i%5000 for i in range(5000)]
grouping = [order.index(id)//50 for id in range(5000)]

head.append('trial_set_id')

new_corpus = []
new_corpus.append(head)
for i, row in enumerate(body):
    row.append(grouping[i])
    new_corpus.append(row)

with open(PATH+'textfile/generated_pairs_with_group.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(new_corpus[0])
    for i in range(1,5001):
        writer.writerow(new_corpus[i])


for group_id in range(100):
    group_list = [i for i,id in enumerate(grouping) if id == group_id]
    verb=[]
    for id in group_list:
        row = new_corpus[id+1]
        verb.append(row[2])

