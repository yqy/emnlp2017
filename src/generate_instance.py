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
        return (pre_zp_x,post_zp_x)


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
