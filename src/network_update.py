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
from conf import *

import lasagne

from NetworkComponet import *

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

'''
Deep neural network for AZP resolution
update embedding when training
'''

#### Constants
GPU = True
if GPU:
    print >> sys.stderr,"Trying to run under a GPU. If this is not desired,then modify NetWork.py\n to set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
        print >> sys.stderr,"Use gpu"
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

        self.w_embedding = init_weight_file(args.embedding,args.embedding_dimention)
        self.params.append(self.w_embedding)

        self.zp_x_pre_index = T.imatrix("zp_x_pre")
        self.zp_x_post_index = T.imatrix("zp_x_post")

        zp_x_pre_newshape = (T.shape(self.zp_x_pre_index)[0],args.embedding_dimention)
        self.embedding_sub_zp_pre = self.w_embedding[self.zp_x_pre_index.flatten()]
        self.zp_x_pre = T.reshape(self.embedding_sub_zp_pre,zp_x_pre_newshape)

        zp_x_post_newshape = (T.shape(self.zp_x_post_index)[0],args.embedding_dimention)
        self.embedding_sub_zp_post = self.w_embedding[self.zp_x_post_index.flatten()]
        self.zp_x_post = T.reshape(self.embedding_sub_zp_post,zp_x_post_newshape)
        
        zp_nn_pre = LSTM(embedding_dimention,n_hidden,self.zp_x_pre)
        self.params += zp_nn_pre.params
        
        zp_nn_post = LSTM(embedding_dimention,n_hidden,self.zp_x_post)
        self.params += zp_nn_post.params

        self.zp_out = T.concatenate((zp_nn_pre.nn_out,zp_nn_post.nn_out))

        self.zp_out_output = self.zp_out

        ### get sequence output for NP ###
        self.np_x_post_index = T.itensor3("np_x")
        self.np_x_postc_index = T.itensor3("np_x")
        self.np_x_pre_index = T.itensor3("np_x")
        self.np_x_prec_index = T.itensor3("np_x")

        np_x_post_newshape = (T.shape(self.np_x_post_index)[0],T.shape(self.np_x_post_index)[1],args.embedding_dimention)
        self.embedding_sub_np_x_post = self.w_embedding[self.np_x_post_index.flatten()]
        self.np_x_post = T.reshape(self.embedding_sub_np_x_post,np_x_post_newshape)

        np_x_postc_newshape = (T.shape(self.np_x_postc_index)[0],T.shape(self.np_x_postc_index)[1],args.embedding_dimention)
        self.embedding_sub_np_x_postc = self.w_embedding[self.np_x_postc_index.flatten()]
        self.np_x_postc = T.reshape(self.embedding_sub_np_x_postc,np_x_postc_newshape)

        np_x_pre_newshape = (T.shape(self.np_x_pre_index)[0],T.shape(self.np_x_pre_index)[1],args.embedding_dimention)
        self.embedding_sub_np_x_pre = self.w_embedding[self.np_x_pre_index.flatten()]
        self.np_x_pre = T.reshape(self.embedding_sub_np_x_pre,np_x_pre_newshape)

        np_x_prec_newshape = (T.shape(self.np_x_prec_index)[0],T.shape(self.np_x_prec_index)[1],args.embedding_dimention)
        self.embedding_sub_np_x_prec = self.w_embedding[self.np_x_prec_index.flatten()]
        self.np_x_prec = T.reshape(self.embedding_sub_np_x_prec,np_x_prec_newshape)

        self.mask_pre = T.matrix("mask")
        self.mask_prec = T.matrix("mask")

        self.mask_post = T.matrix("mask")
        self.mask_postc = T.matrix("mask")
    
        self.np_nn_pre = sub_LSTM_batch(embedding_dimention,n_hidden,self.np_x_pre,self.np_x_prec,self.mask_pre,self.mask_prec)
        self.params += self.np_nn_pre.params
        self.np_nn_post = sub_LSTM_batch(embedding_dimention,n_hidden,self.np_x_post,self.np_x_postc,self.mask_post,self.mask_postc)
        self.params += self.np_nn_post.params

        self.np_nn_post_output = self.np_nn_post.nn_out
        self.np_nn_pre_output = self.np_nn_pre.nn_out

        self.np_out = T.concatenate((self.np_nn_post_output,self.np_nn_pre_output),axis=1)

        np_nn_f = LSTM(n_hidden*2,n_hidden*2,self.np_out)
        self.params += np_nn_f.params
        np_nn_b = LSTM(n_hidden*2,n_hidden*2,self.np_out[::-1])
        self.params += np_nn_b.params

        self.bi_np_out = T.concatenate((np_nn_f.all_hidden,np_nn_b.all_hidden[::-1]),axis=1)

        self.np_out_output = self.bi_np_out
        #self.get_np_out = theano.function(inputs=[self.np_x_pre,self.np_x_prec,self.np_x_post,self.np_x_postc,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc],outputs=[self.np_out_output])

        #self.feature = T.matrix("feature")
        #self.feature_layer = Layer(feature_dimention,n_hidden,self.feature,repre_active) 
        #self.params += self.feature_layer.params

        w_attention_zp,b_attention = init_weight(n_hidden*2,1,pre="attention_zp",ones=False) 
        self.params += [w_attention_zp,b_attention]

        w_attention_np,b_u = init_weight(n_hidden*2,1,pre="attention_np",ones=False) 
        self.params += [w_attention_np]

        w_attention_np_rnn,b_u = init_weight(n_hidden*4,1,pre="attention_np_rnn",ones=False) 
        self.params += [w_attention_np_rnn]

        np_out_dropout = _dropout_from_layer(self.np_out_output) 
        zp_out_dropout = _dropout_from_layer(self.zp_out_output) 
        np_dropout = _dropout_from_layer(self.np_out) 

        self.calcu_attention_dropout = tanh(T.dot(np_out_dropout,w_attention_np_rnn) + T.dot(zp_out_dropout,w_attention_zp) + T.dot(np_dropout,w_attention_np) + b_attention)

        self.calcu_attention = tanh(T.dot(self.np_out_output,w_attention_np_rnn) + T.dot(self.zp_out_output,w_attention_zp) + T.dot(self.np_out,w_attention_np) + b_attention)

        self.attention = softmax(T.transpose(self.calcu_attention,axes=(1,0)))[0]
        self.attention_dropout = softmax(T.transpose(self.calcu_attention_dropout,axes=(1,0)))[0]

        self.out = self.attention
        self.out_dropout = self.attention_dropout

        self.get_out = theano.function(inputs=[self.zp_x_pre_index,self.zp_x_post_index,self.np_x_pre_index,self.np_x_prec_index,self.np_x_post_index,self.np_x_postc_index,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc],outputs=[self.out],on_unused_input='warn')
        
        l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        lmbda_l1 = 0.0
        #lmbda_l2 = 0.001
        lmbda_l2 = 0.0

        t = T.bvector()
        #cost = -(T.log((self.out*t).sum()))
        cost = -(T.log((self.out_dropout*t).sum()))

        lr = T.scalar()
        
        updates = lasagne.updates.sgd(cost, self.params, lr)
        #updates = lasagne.updates.adadelta(cost, self.params)

        
        self.train_step = theano.function(
            inputs=[self.zp_x_pre_index,self.zp_x_post_index,self.np_x_pre_index,self.np_x_prec_index,self.np_x_post_index,self.np_x_postc_index,self.mask_pre,self.mask_prec,self.mask_post,self.mask_postc,t,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def main():
    r = NetWork(3,2)
    t = [0,1,0]
    zp_x = [[2,3],[1,2],[2,3]]

    np_x = [[[1,2],[2,3],[3,1]],[[2,3],[1,2],[2,3]],[[3,3],[1,2],[2,3]]]
    mask = [[1,1,1],[1,1,0],[1,1,1]]
    npp_x = [[[1,2],[2,3]],[[3,3],[2,3]],[[1,1],[2,2]]]
    maskk = [[1,1],[1,0],[0,1]]

    print r.get_np_out(np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    print r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)
    print "Train"
    r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)
    r.train_step(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk,t,5)

    print r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)

    q = list(r.get_out(zp_x,zp_x,np_x,npp_x,np_x,npp_x,mask,maskk,mask,maskk)[0])
    for num in q:
        print num

if __name__ == "__main__":
    main()
