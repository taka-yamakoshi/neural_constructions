import numpy as np
import random
import csv
import sys
sys.path.append('..')
args = sys.argv

def GenerateDO(subject,verb,particle,recipient,theme):
    if particle == '-':
        sentence = f'{subject} {verb_dict[verb]} {recipient} {theme}.'
    else:
        sentence = f'{subject} {verb_dict[verb]} {recipient} {particle} {theme}.'
    return sentence

def GeneratePD(subject,verb,particle,recipient,theme):
    if particle == '-':
        sentence = f'{subject} {verb_dict[verb]} {theme} to {recipient}.'
    else:
        sentence = f'{subject} {verb_dict[verb]} {theme} {particle} to {recipient}.'
    return sentence


with open('../csvfiles/SWBD.csv') as f:
    reader = csv.reader(f)
    file = [row for row in reader]
    head = file[0]
    text = file[1:]

## Create the text file 'VerbPairs.txt', which is already done.
#verb_list = [row[head.index('Verb')] for row in text]
#for word in set(verb_list):
#   print(word)

## Create the text file 'subjects_list.txt' in order to fix the subjects, which is also done.
#subjects = ['Mary', 'Linda', 'Maria', 'Alice', 'John', 'Bob', 'Michael', 'Juan']
#with open('subjects_list.txt','w') as f:
#   for line in text:
#       f.write(random.choice(subjects))
#       f.writelines('\n')
#exit()

with open('VerbPairs.txt','r') as f:
    pairs = f.read().split('\n')

verb_dict = {}
for row in pairs:
    verb_dict[row.split('\t')[0]] = row.split('\t')[1]

new_head = ['sent_id','DOsentence','PDsentence','realized_construction']
with open('subjects_list.txt','r') as f:
    subjects = f.read().split('\n')[:-1]
assert len(subjects) == len(text)

with open('../csvfiles/GeneratedSentsSWBD.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(new_head)
    for id,line in enumerate(text):
        subject = subjects[id]
        verb = line[head.index('Verb')]
        particle = line[head.index('Particle')]
        recipient = line[head.index('Recipient')]
        theme = line[head.index('Theme')]
        if line[head.index('Realized_Construction')] == 'NP NP.':
            writer.writerow([str(id),GenerateDO(subject,verb,particle,recipient,theme), GeneratePD(subject,verb,particle,recipient,theme),'DO'])
        elif line[head.index('Realized_Construction')] == 'NP_PP.':
            writer.writerow([str(id),GenerateDO(subject,verb,particle,recipient,theme), GeneratePD(subject,verb,particle,recipient,theme),'PD'])
        else:
            print('Unrecognized construction type')

