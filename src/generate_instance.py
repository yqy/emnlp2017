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


import cPickle
sys.setrecursionlimit(1000000)


MAX = 2

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

def get_inputs(w2v,nodes_info,sentence_index,begin_index,end_index,ty):
    if ty == "zp":
        ### get for ZP ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        pre_zp_x.append(list([0.0]*args.embedding_dimention))
        for i in range(0,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    pre_zp_x.append(list(em_x))

        post_zp_x = []
        for i in range(end_index+1,len(twl)):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    post_zp_x.append(list(em_x))
        post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x = post_zp_x[::-1]
        return (numpy.array(pre_zp_x,dtype = numpy.float32),numpy.array(post_zp_x,dtype = numpy.float32))

    if ty == "zp_index":
        ### get for ZP ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        #pre_zp_x.append(list([0.0]*args.embedding_dimention))
        pre_zp_x.append([0])
        for i in range(0,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    pre_zp_x.append([em_x])

        post_zp_x = []
        for i in range(end_index+1,len(twl)):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    post_zp_x.append([em_x])
        #post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x.append([0])
        post_zp_x = post_zp_x[::-1]
        return (numpy.array(pre_zp_x,dtype = numpy.int32),numpy.array(post_zp_x,dtype = numpy.int32))


    elif ty == "np":
        tnl,twl = nodes_info[sentence_index]
        np_x_pre = []
        np_x_pre.append(list([0.0]*args.embedding_dimention))
        #for i in range(0,begin_index):
        for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_pre.append(list(em_x))
        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_pre.append(list(em_x))

        np_x_post = []

        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_post.append(list(em_x))

        #for i in range(end_index+1,len(twl)):
        for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_post.append(list(em_x))
        np_x_post.append(list([0.0]*args.embedding_dimention))
        np_x_post = np_x_post[::-1]

        return (np_x_pre,np_x_post)

    elif ty == "npc":
        ### get for NP context ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        pre_zp_x.append(list([0.0]*args.embedding_dimention))
        #for i in range(0,begin_index):
        for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    pre_zp_x.append(list(em_x))

        post_zp_x = []
        #for i in range(end_index+1,len(twl)):
        for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    post_zp_x.append(list(em_x))
        post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x = post_zp_x[::-1]
        
        return pre_zp_x,post_zp_x

    elif ty == "np_unlimited":
        ## unlimited context length
        tnl,twl = nodes_info[sentence_index]
        np_x_pre = []
        np_x_pre.append(list([0.0]*args.embedding_dimention))
        for i in range(0,begin_index):
        #for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_pre.append(list(em_x))
        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_pre.append(list(em_x))

        np_x_post = []

        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_post.append(list(em_x))

        for i in range(end_index+1,len(twl)):
        #for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x_post.append(list(em_x))
        np_x_post.append(list([0.0]*args.embedding_dimention))
        np_x_post = np_x_post[::-1]

        return (np_x_pre,np_x_post)

    elif ty == "npc_unlimited":
        ### get for NP context ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        pre_zp_x.append(list([0.0]*args.embedding_dimention))
        for i in range(0,begin_index):
        #for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    pre_zp_x.append(list(em_x))

        post_zp_x = []
        for i in range(end_index+1,len(twl)):
        #for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    post_zp_x.append(list(em_x))
        post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x = post_zp_x[::-1]
        return (pre_zp_x,post_zp_x)

    elif ty == "np_index":
        ## unlimited context length
        tnl,twl = nodes_info[sentence_index]
        np_x_pre = []
        #np_x_pre.append(list([0.0]*args.embedding_dimention))
        np_x_pre.append([0])
        for i in range(0,begin_index):
        #for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x is not None:
                    np_x_pre.append([em_x])
        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    np_x_pre.append([em_x])

        np_x_post = []

        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x is not None:
                    np_x_post.append([em_x])

        for i in range(end_index+1,len(twl)):
        #for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    np_x_post.append([em_x])
        np_x_post.append([0])
        np_x_post = np_x_post[::-1]
        
        return (np_x_pre,np_x_post)

    elif ty == "npc_index":
        ### get for NP context ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        #pre_zp_x.append(list([0.0]*args.embedding_dimention))
        pre_zp_x.append([0])
        for i in range(0,begin_index):
        #for i in range(begin_index-10,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    pre_zp_x.append([em_x])

        post_zp_x = []
        for i in range(end_index+1,len(twl)):
        #for i in range(end_index+1,end_index+10):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_index_by_word(twl[i].word)
                if em_x >= 0:
                    post_zp_x.append([em_x])
        #post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x.append([0])
        post_zp_x = post_zp_x[::-1]
        
        return pre_zp_x,post_zp_x


def add_mask(np_x_list):
    add_item = list([0.0]*args.embedding_dimention)
    masks = []

    max_len = 0
    for np_x in np_x_list:
        if len(np_x) > max_len:
            max_len = len(np_x)

    for np_x in np_x_list:
        mask = len(np_x)*[1]
        for i in range(max_len-len(np_x)):
            #np_x.append(add_item)
            #mask.append(0)
            np_x.insert(0,add_item)
            mask.insert(0,0)
        masks.append(mask)
    return masks

def add_mask_index(np_x_list):
    add_item = [0]
    masks = []

    max_len = 0
    for np_x in np_x_list:
        if len(np_x) > max_len:
            max_len = len(np_x)

    for np_x in np_x_list:
        mask = len(np_x)*[1]
        for i in range(max_len-len(np_x)):
            #np_x.append(add_item)
            #mask.append(0)
            np_x.insert(0,add_item)
            mask.insert(0,0)
        masks.append(mask)
    return masks


def generate_training_instances(path,w2v):

    paths = get_dir.get_all_file(path,[])
    
    training_instances = []
        
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence
            
            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            res_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)
                    #feature_list.append(ifl)

                    res_list.append(res_result)
            if len(np_x_pre_list) == 0:
                continue
            if sum(res_list) == 0:
                continue

            mask_pre = add_mask(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.float32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.float32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.float32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.float32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            #feature_list = numpy.array(feature_list,dtype = numpy.float32)

            training_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list))
    return training_instances

def generate_test_instances(path,w2v):

    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence


            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            this_nodes_info = {} ## 为了节省存储空间
            np_x_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            res_list = []
            zp_candi_list = [] ## 为了存zp和candidate
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)

                    res_list.append(res_result)
                    zp_candi_list.append((zp,candidate))

                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            if len(np_x_pre_list) == 0:
                continue

            mask_pre = add_mask(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.float32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.float32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.float32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.float32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            #feature_list = numpy.array(feature_list,dtype = numpy.float32)

            anaphorics_result.append(anaphorics)
            test_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,this_nodes_info))
    return test_instances,anaphorics_result

def generate_training_instances_batch(path,w2v):
    ## no mask ##

    paths = get_dir.get_all_file(path,[])
    
    training_instances = []
        
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence
            
            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            res_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    #np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_unlimited")
                    #np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_unlimited")
                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)
                    #feature_list.append(ifl)

                    res_list.append(res_result)
            if len(np_x_pre_list) == 0:
                continue
            if sum(res_list) == 0:
                continue

            training_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,res_list))
    return training_instances

def generate_test_instances_batch(path,w2v):

    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence


            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            this_nodes_info = {} ## 为了节省存储空间
            np_x_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            res_list = []
            zp_candi_list = [] ## 为了存zp和candidate
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    #np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_unlimited")
                    #np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_unlimited")
                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)

                    res_list.append(res_result)
                    zp_candi_list.append((zp,candidate))

                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            if len(np_x_pre_list) == 0:
                continue
            #feature_list = numpy.array(feature_list,dtype = numpy.float32)

            anaphorics_result.append(anaphorics)
            test_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,res_list,zp_candi_list,this_nodes_info))
    return test_instances,anaphorics_result

def generate_batch_instances(raw_instances,typ):
    batch_size = args.batch
    if typ == "train":
        instance_list = []
        instance_batch_list = []

        pos_num = 0
        neg_num = 0

        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,res_list in raw_instances:
            for i in range(len(res_list)):
                if res_list[i] == 1:
                    instance_list.append((zp_x_pre,zp_x_post,np_x_pre_list[i],np_x_prec_list[i],np_x_post_list[i],np_x_postc_list[i],res_list[i])) 
                    pos_num += 1
                else:
                    if pos_num >= 0:
                        instance_list.append((zp_x_pre,zp_x_post,np_x_pre_list[i],np_x_prec_list[i],np_x_post_list[i],np_x_postc_list[i],res_list[i])) 
                        pos_num -= 1
                        neg_num += 1
        
        print >> sys.stderr,"All instance:",len(raw_instances),"pos:",pos_num,"neg:",neg_num," Total batch:",len(raw_instances)/batch_size
        for i in range(len(instance_list)/batch_size):
            this_zp_x_pre = []
            this_zp_x_post = []
            this_np_x_pre_list = []
            this_np_x_prec_list = []
            this_np_x_post_list = []
            this_np_x_postc_list = []
            this_res_list = []
            for zp_x_pre_t,zp_x_post_t,np_x_pre_list_t,np_x_prec_list_t,np_x_post_list_t,np_x_postc_list_t,res_list_t in instance_list[i*batch_size:(i+1)*batch_size]:
                this_zp_x_pre.append(list(zp_x_pre_t))
                this_zp_x_post.append(list(zp_x_post_t))
                this_np_x_pre_list.append(list(np_x_pre_list_t))
                this_np_x_prec_list.append(list(np_x_prec_list_t))
                this_np_x_post_list.append(list(np_x_post_list_t))
                this_np_x_postc_list.append(list(np_x_postc_list_t))
                this_res_list.append(res_list_t)

            mask_zp_x_pre = add_mask(this_zp_x_pre) 
            this_zp_x_pre = numpy.array(this_zp_x_pre,dtype = numpy.float32)
            mask_zp_x_pre = numpy.array(mask_zp_x_pre,dtype = numpy.float32)

            mask_zp_x_post = add_mask(this_zp_x_post) 
            this_zp_x_post = numpy.array(this_zp_x_post,dtype = numpy.float32)
            mask_zp_x_post = numpy.array(mask_zp_x_post,dtype = numpy.float32)

            mask_np_x_pre_list = add_mask(this_np_x_pre_list) 
            this_np_x_pre_list = numpy.array(this_np_x_pre_list,dtype = numpy.float32)
            mask_np_x_pre_list = numpy.array(mask_np_x_pre_list,dtype = numpy.float32)

            mask_np_x_prec_list = add_mask(this_np_x_prec_list) 
            this_np_x_prec_list = numpy.array(this_np_x_prec_list,dtype = numpy.float32)
            mask_np_x_prec_list = numpy.array(mask_np_x_prec_list,dtype = numpy.float32)

            mask_np_x_post_list = add_mask(this_np_x_post_list) 
            this_np_x_post_list = numpy.array(this_np_x_post_list,dtype = numpy.float32)
            mask_np_x_post_list = numpy.array(mask_np_x_post_list,dtype = numpy.float32)

            mask_np_x_postc_list = add_mask(this_np_x_postc_list) 
            this_np_x_postc_list = numpy.array(this_np_x_postc_list,dtype = numpy.float32)
            mask_np_x_postc_list = numpy.array(mask_np_x_postc_list,dtype = numpy.float32)

            instance_batch_list.append((this_zp_x_pre,this_zp_x_post,mask_zp_x_pre,mask_zp_x_post,this_np_x_pre_list,this_np_x_prec_list,this_np_x_post_list,this_np_x_postc_list,mask_np_x_pre_list,mask_np_x_prec_list,mask_np_x_post_list,mask_np_x_postc_list,this_res_list)) 
        return instance_batch_list

    elif typ == "test":
        instance_batch_list = []
        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,res_list,zp_candi_list,this_nodes_inf in raw_instances:

            this_np_x_pre_list = list(np_x_pre_list)
            this_np_x_prec_list = list(np_x_prec_list)
            this_np_x_post_list = list(np_x_post_list)
            this_np_x_postc_list = list(np_x_postc_list)

            mask_pre = add_mask(this_np_x_pre_list) 
            this_np_x_pre_list = numpy.array(this_np_x_pre_list,dtype = numpy.float32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask(this_np_x_prec_list) 
            this_np_x_prec_list = numpy.array(this_np_x_prec_list,dtype = numpy.float32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask(this_np_x_post_list) 
            this_np_x_post_list = numpy.array(this_np_x_post_list,dtype = numpy.float32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask(this_np_x_postc_list) 
            this_np_x_postc_list = numpy.array(this_np_x_postc_list,dtype = numpy.float32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            this_zp_x_pre = [list(zp_x_pre)]*len(res_list)
            this_zp_x_post = [list(zp_x_post)]*len(res_list)

            mask_zp_x_pre = add_mask(this_zp_x_pre) 
            this_zp_x_pre = numpy.array(this_zp_x_pre,dtype = numpy.float32)
            mask_zp_x_pre = numpy.array(mask_zp_x_pre,dtype = numpy.float32)

            mask_zp_x_post = add_mask(this_zp_x_post) 
            this_zp_x_post = numpy.array(this_zp_x_post,dtype = numpy.float32)
            mask_zp_x_post = numpy.array(mask_zp_x_post,dtype = numpy.float32)

            instance_batch_list.append((this_zp_x_pre,this_zp_x_post,mask_zp_x_pre,mask_zp_x_post,this_np_x_pre_list,this_np_x_prec_list,this_np_x_post_list,this_np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,this_nodes_inf)) 
        return instance_batch_list


def generate_training_instances_index(path,w2v):

    paths = get_dir.get_all_file(path,[])
    
    training_instances = []
        
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence
            
            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp_index")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            res_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_index")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_index")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)
                    #feature_list.append(ifl)

                    res_list.append(res_result)
            if len(np_x_pre_list) == 0:
                continue
            if sum(res_list) == 0:
                continue

            mask_pre = add_mask_index(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.int32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask_index(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.int32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask_index(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.int32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask_index(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.int32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            #feature_list = numpy.array(feature_list,dtype = numpy.float32)

            training_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list))
    return training_instances

def generate_test_instances_index(path,w2v):

    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence


            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp_index")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            this_nodes_info = {} ## 为了节省存储空间
            np_x_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            res_list = []
            zp_candi_list = [] ## 为了存zp和candidate
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_index")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_index")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)

                    res_list.append(res_result)
                    zp_candi_list.append((zp,candidate))

                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            if len(np_x_pre_list) == 0:
                continue

            mask_pre = add_mask_index(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.int32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask_index(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.int32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask_index(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.int32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask_index(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.int32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            #feature_list = numpy.array(feature_list,dtype = numpy.float32)

            anaphorics_result.append(anaphorics)
            test_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,this_nodes_info))
    return test_instances,anaphorics_result

def generate_training_instances_feature(path,w2v):

    paths = get_dir.get_all_file(path,[])
    HcP = []
    
    training_instances = []
        
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence
            
            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            res_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    ifl = get_feature.get_res_feature_NN_new(zp,candidate,zp_wl,candi_wl)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)

                    feature_list.append(ifl)

                    res_list.append(res_result)
            if len(np_x_pre_list) == 0:
                continue
            if sum(res_list) == 0:
                continue

            mask_pre = add_mask(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.float32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.float32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.float32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.float32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            feature_list = numpy.array(feature_list,dtype = numpy.float32)

            training_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list))
    return training_instances

def generate_test_instances_feature(path,w2v):

    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    HcP = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence


            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            this_nodes_info = {} ## 为了节省存储空间
            np_x_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            res_list = []
            zp_candi_list = [] ## 为了存zp和candidate
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    ifl = get_feature.get_res_feature_NN_new(zp,candidate,zp_wl,candi_wl)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)

                    res_list.append(res_result)
                    feature_list.append(ifl)

                    zp_candi_list.append((zp,candidate))

                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            if len(np_x_pre_list) == 0:
                continue

            mask_pre = add_mask(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.float32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.float32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.float32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.float32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            feature_list = numpy.array(feature_list,dtype = numpy.float32)

            anaphorics_result.append(anaphorics)
            test_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,zp_candi_list,this_nodes_info))
    return test_instances,anaphorics_result

def generate_training_instances_feature_update(path,w2v):

    paths = get_dir.get_all_file(path,[])
    
    training_instances = []
        
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence
            
            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp_index")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            res_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_index")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_index")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    ifl = get_feature.get_res_feature_NN_new(zp,candidate,zp_wl,candi_wl)                   
 
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)
                    feature_list.append(ifl)

                    res_list.append(res_result)
            if len(np_x_pre_list) == 0:
                continue
            if sum(res_list) == 0:
                continue

            mask_pre = add_mask_index(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.int32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask_index(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.int32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask_index(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.int32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask_index(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.int32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            feature_list = numpy.array(feature_list,dtype = numpy.float32)

            training_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list))
    return training_instances

def generate_test_instances_feature_update(path,w2v):

    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
            for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence


            zp = (sentence_index,zp_index)
            zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp_index")

            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0
            this_nodes_info = {} ## 为了节省存储空间
            np_x_list = []
            np_x_pre_list = []
            np_x_prec_list = []
            np_x_post_list = []
            np_x_postc_list = []
            res_list = []
            zp_candi_list = [] ## 为了存zp和candidate
            feature_list = []

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    np_x_pre,np_x_post = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np_index")
                    np_x_prec,np_x_postc = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"npc_index")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    if len(np_x_pre) == 0:
                        continue
                    ifl = get_feature.get_res_feature_NN_new(zp,candidate,zp_wl,candi_wl)
                    
                    np_x_pre_list.append(np_x_pre)
                    np_x_prec_list.append(np_x_prec)
                    np_x_post_list.append(np_x_post)
                    np_x_postc_list.append(np_x_postc)
                    feature_list.append(ifl)

                    res_list.append(res_result)
                    zp_candi_list.append((zp,candidate))

                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]
                    

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            if len(np_x_pre_list) == 0:
                continue

            mask_pre = add_mask_index(np_x_pre_list) 
            np_x_pre_list = numpy.array(np_x_pre_list,dtype = numpy.int32)
            mask_pre = numpy.array(mask_pre,dtype = numpy.float32)

            mask_prec = add_mask_index(np_x_prec_list) 
            np_x_prec_list = numpy.array(np_x_prec_list,dtype = numpy.int32)
            mask_prec = numpy.array(mask_prec,dtype = numpy.float32)

            mask_post = add_mask_index(np_x_post_list) 
            np_x_post_list = numpy.array(np_x_post_list,dtype = numpy.int32)
            mask_post = numpy.array(mask_post,dtype = numpy.float32)

            mask_postc = add_mask_index(np_x_postc_list) 
            np_x_postc_list = numpy.array(np_x_postc_list,dtype = numpy.int32)
            mask_postc = numpy.array(mask_postc,dtype = numpy.float32)

            feature_list = numpy.array(feature_list,dtype = numpy.float32)

            anaphorics_result.append(anaphorics)
            test_instances.append((zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,zp_candi_list,this_nodes_info))
    return test_instances,anaphorics_result

