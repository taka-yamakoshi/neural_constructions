import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
args = sys.argv
with open("mypath.txt") as f:
    PATH = f.read()

with open(PATH + 'datafile/train_performance.pkl','rb') as f:
    train_performance = pickle.load(f)
with open(PATH + 'datafile/test_performance.pkl','rb') as f:
    test_performance = pickle.load(f)

plt.imshow(train_performance)
plt.xlabel("Head Number")
plt.ylabel("Layer Number")
plt.show()
plt.imshow(test_performance)
plt.xlabel("Head Number")
plt.ylabel("Layer Number")
plt.show()
