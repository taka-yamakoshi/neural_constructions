import pickle
import sys
sys.path.append('..')
with open("mypath.txt") as f:
    PATH = f.read()
args = sys.argv

good_verbs = ["brought","took","carried","dragged","pulled","pushed","phoned","telephoned","fed","gave","leased","lent","loaned","passed","paid","rented","repaid","sold","served","allocated","assigned","awarded","ceded","conceded","extended","granted","guaranteed","issued","offered","owed","promised","voted","yielded","forwarded","handed","mailed","posted","sent","shipped","slipped","smuggled","sneaked","pitched","threw","tipped","tossed","asked","read","relayed","showed","taught","told","wrote"]
bad_verbs = ["dropped","credited","furnished","presented","provided","supplied","trusted","addressed","administered","broadcast","conveyed","contributed","delegated","delivered","donated","exhibited","expressed","explained","illustrated","introduced","narrated","recited","recommended","referred","reimbursed","restored","returned","sacrificed","submitted","surrendered","transferred","transported","admitted","alleged","announced","articulated","asserted","communicated","confessed","declared","mentioned","proposed","recounted","repeated","reported","revealed","said","stated","cried","raged","screamed","shouted","sang","whispered","yelled"]
verbs = [good_verbs,bad_verbs]
themes = ["box","news","food","money","land","salt","dish","task","support","warning","deal","ballot","letter","package","ball","question","story","message","lesson","award","fund","plan","medicine","police","evidence","emotion","problem","poem","book","friend","work","order","song","secret"]

verb_types = ["good","bad"]
for i, verb_type in enumerate(verb_types):
    with open(PATH+"textfile/verb_theme_pairs_"+verb_type+"_all.txt",'w') as f:
        for verb in verbs[i]:
            for theme in themes:
                f.write(verb+" "+theme)
                f.writelines('\n')
