#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
from subprocess import *
random.seed(110)

from conf import *
from buildTree import get_info_from_file
from buildTree import get_info_from_file_system
import get_dir
import get_feature
import word2vec
#import network
#import network_pair as network
#import network_single as network
#import network_update as network
#import network_selfAttention as network
#import network_attention as network
#import network_feature as network
import network_feature_update as network
import generate_instance


import cPickle
sys.setrecursionlimit(1000000)

if(len(sys.argv) <= 1): 
    sys.stderr.write("Not specify options, type '-h' for help\n")
    exit()

print >> sys.stderr, os.getpid()

def get_prf(anaphorics_result,predict_result):
    ## 如果 ZP 是负例 则没有anaphorics_result
    ## 如果 predict 出负例 则 predict_candi_sentence_index = -1
    should = 0
    right = 0
    predict_right = 0
    for i in range(len(predict_result)):
        (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) = predict_result[i]
        anaphoric = anaphorics_result[i] 
        if anaphoric:
            should += 1
            if (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) in anaphoric:
                right += 1
        if not (predict_candi_sentence_index == -1):
            predict_right += 1

    print "Should:",should,"Right:",right,"PredictRight:",predict_right
    if predict_right == 0:
        P = 0.0
    else:
        P = float(right)/float(predict_right)

    if should == 0:
        R = 0.0
    else:
        R = float(right)/float(should)

    if (R == 0.0) or (P == 0.0):
        F = 0.0
    else:
        F = 2.0/(1.0/P + 1.0/R)

    print "P:",P
    print "R:",R
    print "F:",F


def get_sentence(zp_sentence_index,zp_index,nodes_info):
    #返回只包含zp_index位置的ZP的句子
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"): 
                return_words.append(this_word)
    return " ".join(return_words)

def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info

def find_max(l):
    ### 找到list中最大的 返回index
    return_index = len(l)-1
    max_num = 0.0
    for i in range(len(l)):
        if l[i] >= max_num:
            max_num = l[i] 
            return_index = i
    return return_index

def find_max_batch(l):
    ### 找到list中最大的 返回index
    return_index = len(l)-1
    max_num = 0.0
    for i in range(len(l)):
        if l[i][1] >= max_num:
            max_num = l[i][1]
            return_index = i
    return return_index


def get_dijian(n):
    norm = 0.5
    p_list = [0.7,0.7,0.8,1.0]
    if n <= 4:
        return numpy.array(p_list[-n:])
    else:
        return numpy.array([0.5]*(n-4)+p_list)

if args.type == "nn_train":

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2Vec(args.embedding)
        
        ### Training ####    
        path = args.data
        training_instances = generate_instance.generate_training_instances(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        test_instances,anaphorics_result = generate_instance.generate_test_instances(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        LSTM = network.NetWork(100,args.embedding_dimention,61)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(training_instances)

        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in training_instances:
            #np_num = len(res_list)
            #dijian = get_dijian(np_num)
            cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,args.lr)[0]
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,dijian,args.lr)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"


        '''
        ### see how many hts ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x,mask,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x,mask)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Training Hits:",hits,"/",len(training_instances)
        '''

        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        hits = 0
        for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,nodes_info) in test_instances:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc)[0])
                max_index = find_max(outputs)
                if res_list[max_index] == 1:
                    hits += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))

        print >> sys.stderr, "Test Hits:",hits,"/",len(test_instances)

        '''
        ### see how many hits in DEV ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Dev Hits:",hits,"/",len(training_instances)
        '''
        print "Echo",echo 
        print "Test Hits:",hits,"/",len(test_instances)
        get_prf(anaphorics_result,predict_result)

        sys.stdout.flush()
    print >> sys.stderr,"Over for all"

if args.type == "nn_single":
    ##每次放一对训练，用batch

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2Vec(args.embedding)
        
        ### Training ####    
        path = args.data
        training_instances = generate_instance.generate_training_instances_batch(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        test_instances,anaphorics_result = generate_instance.generate_test_instances_batch(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    print >> sys.stderr,"Begin generate batch instance"
    start_time = timeit.default_timer()
    training_instances_batch = generate_instance.generate_batch_instances(training_instances,"train")
    test_instances_batch = generate_instance.generate_batch_instances(test_instances,"test")
    end_time = timeit.default_timer()
    print >> sys.stderr, "batch cost",end_time - start_time, "seconds!"


    training_instances = []
    test_instances = []

    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        #LSTM = network.NetWork(100,args.embedding_dimention,61)
        LSTM = network.NetWork(128,args.embedding_dimention,61)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(training_instances_batch)

        for zp_x_pre,zp_x_post,zp_x_pre_mask,zp_x_post_mask,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in training_instances_batch:
            #np_num = len(res_list)
            #dijian = get_dijian(np_num)
            cost += LSTM.train_step(zp_x_pre,zp_x_post,zp_x_pre_mask,zp_x_post_mask,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,args.lr)[0]
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,dijian,args.lr)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"


        '''
        ### see how many hts ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x,mask,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x,mask)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Training Hits:",hits,"/",len(training_instances)
        '''

        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        hits = 0
        for (zp_x_pre,zp_x_post,zp_x_pre_mask,zp_x_post_mask,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,nodes_info) in test_instances_batch:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,zp_x_pre_mask,zp_x_post_mask,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc)[0])
                max_index = find_max_batch(outputs)
                if res_list[max_index] == 1:
                    hits += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i][1]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances_batch))

        print >> sys.stderr, "Test Hits:",hits,"/",len(test_instances_batch)

        '''
        ### see how many hits in DEV ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Dev Hits:",hits,"/",len(training_instances)
        '''


        print "Echo",echo 
        print "Test Hits:",hits,"/",len(test_instances_batch)
        get_prf(anaphorics_result,predict_result)


        sys.stdout.flush()
    print >> sys.stderr,"Over for all"


if args.type == "nn_update":

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2VecIndex(args.embedding)
        
        ### Training ####    
        path = args.data
        training_instances = generate_instance.generate_training_instances_index(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        test_instances,anaphorics_result = generate_instance.generate_test_instances_index(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        LSTM = network.NetWork(100,args.embedding_dimention,61)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(training_instances)

        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in training_instances:
            #np_num = len(res_list)
            #dijian = get_dijian(np_num)
            cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,args.lr)[0]
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,dijian,args.lr)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"


        '''
        ### see how many hts ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x,mask,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x,mask)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Training Hits:",hits,"/",len(training_instances)
        '''

        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        hits = 0
        for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,nodes_info) in test_instances:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc)[0])
                max_index = find_max(outputs)
                if res_list[max_index] == 1:
                    hits += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))

        print >> sys.stderr, "Test Hits:",hits,"/",len(test_instances)

        '''
        ### see how many hits in DEV ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Dev Hits:",hits,"/",len(training_instances)
        '''
        print "Echo",echo 
        print "Test Hits:",hits,"/",len(test_instances)
        get_prf(anaphorics_result,predict_result)

        sys.stdout.flush()
    print >> sys.stderr,"Over for all"

if args.type == "nn_feature":

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2Vec(args.embedding)
        
        ### Training ####    
        path = args.data
        training_instances = generate_instance.generate_training_instances_feature(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        test_instances,anaphorics_result = generate_instance.generate_test_instances_feature(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        LSTM = network.NetWork(128,args.embedding_dimention,61)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(training_instances)

        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list in training_instances:
            #np_num = len(res_list)
            #dijian = get_dijian(np_num)
            cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,args.lr)[0]
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,dijian,args.lr)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"


        '''
        ### see how many hts ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x,mask,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x,mask)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Training Hits:",hits,"/",len(training_instances)
        '''

        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        hits = 0
        for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,zp_candi_list,nodes_info) in test_instances:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list)[0])
                max_index = find_max(outputs)
                if res_list[max_index] == 1:
                    hits += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))

        print >> sys.stderr, "Test Hits:",hits,"/",len(test_instances)

        '''
        ### see how many hits in DEV ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Dev Hits:",hits,"/",len(training_instances)
        '''
        print "Echo",echo 
        print "Test Hits:",hits,"/",len(test_instances)
        get_prf(anaphorics_result,predict_result)

        sys.stdout.flush()
    print >> sys.stderr,"Over for all"

if args.type == "nn_feature_update":

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2VecIndex(args.embedding)        
        
        ### Training ####    
        path = args.data
        training_instances = generate_instance.generate_training_instances_feature_update(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        test_instances,anaphorics_result = generate_instance.generate_test_instances_feature_update(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        LSTM = network.NetWork(128,args.embedding_dimention,84)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(training_instances)

        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list in training_instances:
            #np_num = len(res_list)
            #dijian = get_dijian(np_num)
            cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,args.lr)[0]
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,dijian,args.lr)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"


        '''
        ### see how many hts ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x,mask,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x,mask)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Training Hits:",hits,"/",len(training_instances)
        '''

        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        hits = 0
        for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,zp_candi_list,nodes_info) in test_instances:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list)[0])
                max_index = find_max(outputs)
                if res_list[max_index] == 1:
                    hits += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))

        print >> sys.stderr, "Test Hits:",hits,"/",len(test_instances)

        '''
        ### see how many hits in DEV ###
        hits = 0
        for zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list,res_list in training_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_list,np_x_pre_list,np_x_post_list,mask,mask_pre,mask_post,feature_list)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                hits += 1 
        print >> sys.stderr, "Dev Hits:",hits,"/",len(training_instances)
        '''
        print "Echo",echo 
        print "Test Hits:",hits,"/",len(test_instances)
        get_prf(anaphorics_result,predict_result)

        sys.stdout.flush()
    print >> sys.stderr,"Over for all"


