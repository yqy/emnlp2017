#coding=utf8
import sys
import gzip
import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

from theano.compile.nanguardmode import NanGuardMode


import lasagne

from NetworkComponet import *

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

'''
Deep neural network for AZP resolution
每次放入一对 zp-np pair
加入batch
'''

#### Constants
GPU = True
if GPU:
    print >> sys.stderr,"Trying to run under a GPU. If this is not desired,then modify NetWork.py\n to set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set 
    theano.config.floatX = 'float32'
else:
    print >> sys.stderr,"Running with a CPU. If this is not desired,then modify the \n NetWork.py to set\nthe GPU flag to True."
    theano.config.floatX = 'float64'
class NetWork():
    def __init__(self,n_hidden,embedding_dimention=50,feature_dimention=61):

        ##n_in: sequence lstm 的输入维度
        ##n_hidden: lstm for candi and zp 的隐层维度

        self.params = []

        self.zp_x_pre = T.tensor3("zp_x_pre")
        self.zp_x_post = T.tensor3("zp_x_post")
        self.zp_mask_pre = T.matrix("zp_mask_pre")
        self.zp_mask_post = T.matrix("zp_mask_post")
        
        zp_nn_pre = LSTM_batch(embedding_dimention,n_hidden,self.zp_x_pre,self.zp_mask_pre,"zp_pre_")
        self.params += zp_nn_pre.params
        
        zp_nn_post = LSTM_batch(embedding_dimention,n_hidden,self.zp_x_post,self.zp_mask_post,"zp_post_")
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out),axis=1)
        self.get_zp_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post],outputs=[self.zp_out])

        ### get sequence output for NP ###
        self.np_x_post = T.tensor3("np_x")
        self.np_x_postc = T.tensor3("np_x")

        self.np_x_pre = T.tensor3("np_x")
        self.np_x_prec = T.tensor3("np_x")

        self.mask_pre = T.matrix("mask")
        self.mask_prec = T.matrix("mask")

        self.mask_post = T.matrix("mask")
        self.mask_postc = T.matrix("mask")
    
        np_nn_pre = sub_LSTM_batch(embedding_dimention,n_hidden,self.np_x_pre,self.np_x_prec,self.mask_pre,self.mask_prec,"np_pre_")
        self.params += np_nn_pre.params
        np_nn_post = sub_LSTM_batch(embedding_dimention,n_hidden,self.np_x_post,self.np_x_postc,self.mask_post,self.mask_postc,"np_post_")
        self.params += np_nn_post.params

        self.np_nn_post_output = np_nn_post.nn_out
        self.np_nn_pre_output = np_nn_pre.nn_out
        self.np_out = T.concatenate((self.np_nn_post_output,self.np_nn_pre_output),axis=1)

        self.get_np_out = theano.function(inputs=[self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc],outputs=[self.np_out])

        w_attention_zp,b_attention = init_weight(n_hidden*2,2,pre="attention_zp",ones=False) 
        self.params += [w_attention_zp,b_attention]

        w_attention_np,b_u = init_weight(n_hidden*2,2,pre="attention_np",ones=False) 
        self.params += [w_attention_np]

        self.calcu_attention = tanh(T.dot(self.zp_out,w_attention_zp) + T.dot(self.np_out,w_attention_np) + b_attention)
        self.score = softmax(self.calcu_attention)

        #self.get_attention = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc],outputs=[self.score])
        self.get_out = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc],outputs=[self.score])

        t = T.bvector()
        cost = -( T.log(self.score)[T.arange(t.shape[0]), t] ).mean()
        #cost = T.log(self.score)[T.arange(t.shape[0]), t]

        self.get_cost = theano.function(inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,t],outputs=[cost])

        lr = T.scalar()
        
        updates = lasagne.updates.sgd(cost, self.params, lr)
        #updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(
            inputs=[self.zp_x_pre,self.zp_x_post,self.zp_mask_pre,self.zp_mask_post,self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,t,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(4,2)
    t = [1,0,1]
    zp_x = [[[2,3],[1,2],[2,3],[1,5]],[[2,3],[1,1],[3,2],[3,2]],[[-2,-3],[12,-1],[-1,-2],[-3,-2]]]
    zp_mask = [[1,1,1,1],[1,1,0,1],[1,0,0,1]]

    np_x = [[[1,2],[2,3],[3,1]],[[2,3],[1,2],[2,3]],[[3,3],[1,2],[2,3]]]
    mask = [[1,1,1],[1,1,0],[1,1,1]]
    npp_x = [[[1,2],[2,3]],[[3,3],[2,3]],[[1,1],[2,2]]]
    maskk = [[1,1],[1,0],[0,1]]

    print r.get_zp_out(zp_x,zp_x,zp_mask,zp_mask)
    print r.get_np_out(np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    print r.get_out(zp_x,zp_x,zp_mask,zp_mask,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)

    print "Train"
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    print r.train_step(zp_x,zp_x,zp_mask,zp_mask,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)

    print r.get_out(zp_x,zp_x,zp_mask,zp_mask,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)

    #print r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    #print r.get_max(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t)
    #print "Train"
    #print r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    #print r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    #print r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)

    #print r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)

    #q = list(r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)[0])
    #for num in q:
    #    print num

if __name__ == "__main__":
    main()
