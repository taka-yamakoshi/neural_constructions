import pickle
import sys
sys.path.append('..')
with open("mypath.txt") as f:
    PATH = f.read()
args = sys.argv

#Load verb_theme_pairs
file = open(PATH +'textfile/verb_theme_pairs_good.txt')
verb_theme_pairs_good = file.read().split('\n')[:-1]
file.close()

file = open(PATH +'textfile/verb_theme_pairs_bad.txt')
verb_theme_pairs_bad = file.read().split('\n')[:-1]
file.close()

pairs = [verb_theme_pairs_good,verb_theme_pairs_bad]

verb_type = ['good','bad']
type = ['do','pd']
for i in range(2):
    for j in range(2):
        with open(PATH +'textfile/'+ verb_type[i] + '_'+type[j]+'test.txt','w') as f:
            if j == 0:
                for pair in pairs[i]:
                    f.write("[CLS] the man ")
                    f.write(pair.split(" ")[0])
                    f.write(" her the ")
                    f.write(pair.split(" ")[1])
                    f.write(" . [SEP]")
                    f.writelines('\n')
                    f.write("[CLS] the woman ")
                    f.write(pair.split(" ")[0])
                    f.write(" him the ")
                    f.write(pair.split(" ")[1])
                    f.write(" . [SEP]")
                    f.writelines('\n')
            if j == 1:
                for pair in pairs[i]:
                    f.write("[CLS] the man ")
                    f.write(pair.split(" ")[0])
                    f.write(" the ")
                    f.write(pair.split(" ")[1])
                    f.write(" to ")
                    f.write("her . [SEP]")
                    f.writelines('\n')
                    f.write("[CLS] the woman ")
                    f.write(pair.split(" ")[0])
                    f.write(" the ")
                    f.write(pair.split(" ")[1])
                    f.write(" to ")
                    f.write("him . [SEP]")
                    f.writelines('\n')
