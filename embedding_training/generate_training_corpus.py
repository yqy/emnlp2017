import sys
import re
voc_file = sys.argv[1].strip()
voc_list = []
f = open(voc_file)
while True:
    line = f.readline()
    if not line:break
    line = line.strip().split(" ")
    voc_list.append(line[0].strip())

corpus_file = sys.argv[2].strip()
f = open(corpus_file)
while True:
    line = f.readline()
    if not line:break
    line = re.sub("\s+"," ",line.strip())
    line = line.split(" ")
    for i in range(2,len(line)-2):
        #if line[i] in voc_list:
        print line[i-2],line[i-1],line[i],line[i+1],line[i+2]
            
    
