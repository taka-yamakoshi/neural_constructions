import numpy as np
import pickle
import torch
import sys
sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

names = ["mohamed","sara","liam","maria","ali","adam","lucas","emma","oliver","olivia"]
with open(PATH+"textfile/verb_theme_pairs_good.txt",'r') as f:
    verb_theme_pairs_good = f.read().split('\n')
verb_good = [pair.split(" ")[0] for pair in verb_theme_pairs_good]
theme_good = [pair.split(" ")[1] for pair in verb_theme_pairs_good]
with open(PATH+"textfile/verb_theme_pairs_bad.txt",'r') as f:
    verb_theme_pairs_bad = f.read().split('\n')
verb_bad = [pair.split(" ")[0] for pair in verb_theme_pairs_bad]
theme_bad = [pair.split(" ")[1] for pair in verb_theme_pairs_bad]
