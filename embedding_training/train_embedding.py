#coding=utf8
import sys
import timeit
import random
import numpy

import neural
from conf import *
import wordAttr

def get_index(d,w):
    if w in d:
        return d[w]
    else:
        return 0

def main():

    embedding = []
    start_time = timeit.default_timer()
   
    print >> sys.stderr,"Read word list from embdding"     

    word_dict = {}
    word_file = open(args.embedding_file)
    line = word_file.readline()
    # word_dict[UNK] = 0
    # embedding 第0维: UNK
    index = 1 
    while True:
        line = word_file.readline()
        if not line:break
        line = line.strip().split(" ")[0]
        word_dict[line] = index
        index += 1
    WA = wordAttr.wordAttrDict("./zh-attributes.data")

    ## generate training data ##
    training_instances = []
    f = open(args.data)
    word_type_calculate = {}
    while True:
        line = f.readline()
        if not line:break
        line = line.strip().split(" ")
        if not len(line) == 5:
            continue
        center_word = line[2]
        if not center_word in word_dict:
            continue
        y1,y2,y3 = WA.get_att(center_word)
        word_type_calculate.setdefault((y1,y2,y3),0)
        word_type_calculate[(y1,y2,y3)] += 1

        y = word_dict[center_word]
        #if (line[0] in word_dict) and (line[1] in word_dict) and (line[3] in word_dict) and (line[4] in word_dict):
            #training_instances.append( ([word_dict[line[0]],word_dict[line[1]],word_dict[line[3]],word_dict[line[4]]],y,y1,y2,y3) )
        training_instances.append( ([get_index(word_dict,line[0]),get_index(word_dict,line[1]), get_index(word_dict,line[3]), get_index(word_dict,line[4])],y,y1,y2,y3) )

    print >> sys.stderr, "Generate totally",len(training_instances),"instances"
    end_time = timeit.default_timer()
    print >> sys.stderr,"Totally", end_time - start_time, "seconds!"
    
    for k in sorted(word_type_calculate.keys(),key=lambda a:word_type_calculate[a], reverse = True):
        print >> sys.stderr, k, word_type_calculate[k]

    training_instances_batch = []
    for i in range(len(training_instances)/args.batch_size):
        xl = []
        yl = []
        y1l = []
        y2l = []
        y3l = []
        for x,y,y1,y2,y3 in training_instances[i*args.batch_size:(i+1)*args.batch_size]:
            xl.append(x)
            yl.append(y)
            y1l.append(y1)
            y2l.append(y2)
            y3l.append(y3)
        training_instances_batch.append((numpy.array(xl,dtype=numpy.int32),numpy.array(yl,dtype=numpy.int32),numpy.array(y1l,dtype=numpy.int32),numpy.array(y2l,dtype=numpy.int32),numpy.array(y3l,dtype=numpy.int32)) ) 

    print >> sys.stderr, "Building neural network"
    net = neural.Net(args.embedding_dimention,100,len(word_dict),args.batch_size)    
    random.shuffle(training_instances_batch)

    print >> sys.stderr, "Begin Train!"
    loss = 999999999999
    for i in range(args.echos):
        this_loss = 0.0
        for x,y,y1,y2,y3 in training_instances_batch:
            this_loss += net.train_step(x,y,y1,y2,y3,args.lr)[0]
        if this_loss  < loss:
            loss = this_loss
            embedding = net.show_para()
            fw = open("./embedding","w")
            for em in embedding:
                fw.write(str(list(em)))
                fw.write("\n")
            fw.close()
        end_time = timeit.default_timer()
        print >> sys.stderr,"One Epoch for: ", end_time - start_time, "seconds!"
        print >> sys.stderr,"Loss:",this_loss


if __name__ == "__main__":
    main()
