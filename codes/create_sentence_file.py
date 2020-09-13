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
recipients = ["him","her","the man", "the woman","a man","a woman","the man from childhood","the woman from childhood","a man from childhood","a woman from childhood"]
names = ["mohamed","sara","liam","maria","ali","adam","lucas","emma","oliver","olivia"]

verb_type = ['good','bad']
type = ['do','pd']
for i in range(2):
    for j in range(2):
        with open(PATH +'textfile/'+ verb_type[i] + '_'+type[j]+'_test.txt','w') as f:
            if j == 0:
                for pair in pairs[i]:
                    for recipient in recipients:
                        for subj in names:
                            f.write("[CLS] "+subj+" ")
                            f.write(pair.split(" ")[0])
                            f.write(" "+recipient+" the ")
                            f.write(pair.split(" ")[1])
                            f.write(" . [SEP]")
                            f.writelines('\n')
            if j == 1:
                for pair in pairs[i]:
                    for recipient in recipients:
                        for subj in names:
                            f.write("[CLS] "+subj+" ")
                            f.write(pair.split(" ")[0])
                            f.write(" the ")
                            f.write(pair.split(" ")[1])
                            f.write(" to ")
                            f.write(recipient + " . [SEP]")
                            f.writelines('\n')
