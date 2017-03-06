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

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

#aaaaa

#activation function
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants

def init_weight(n_in,n_out,activation_fn=sigmoid,pre="",uni=True,ones=False):
    rng = np.random.RandomState(1234)
    if uni:
        W_values = np.asarray(rng.normal(size=(n_in, n_out), scale= .01, loc = .0), dtype = theano.config.floatX)
    else:
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / np.sqrt(n_in + n_out)),
                high=np.sqrt(6. / np.sqrt(n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        if activation_fn == theano.tensor.nnet.sigmoid:
            W_values *= 4
            W_values /= 6

    b_values = np.zeros((n_out,), dtype=theano.config.floatX)

    if ones:
        b_values = np.ones((n_out,), dtype=theano.config.floatX)

    w = theano.shared(
        value=W_values,
        name='%sw'%pre, borrow=True
    )
    b = theano.shared(
        value=b_values,
        name='%sb'%pre, borrow=True
    )
    return w,b

class Layer():
    def __init__(self,n_in,n_out,inpt,activation_fn=tanh):
        self.params = []
        if inpt:
            self.inpt = inpt
        else:
            self.inpt= T.matrix("inpt")
        self.w,self.b = init_weight(n_in,n_out,pre="MLP_")
        self.params.append(self.w) 
        self.params.append(self.b) 
    
        self.output = activation_fn(T.dot(self.inpt, self.w) + self.b)

class LinearLayer():
    def __init__(self,n_in,n_out,inpt,activation_fn=tanh):
        self.params = []
        if inpt:
            self.inpt = inpt
        else:
            self.inpt= T.matrix("inpt")
        self.w,self.b = init_weight(n_in,n_out,pre="MLP_")
        self.params.append(self.w) 
        #self.params.append(self.b) 
    
        self.output = T.dot(self.inpt, self.w)

def _dropout_from_layer(layer, p=0.5):
    """p is the probablity of dropping a unit
    """
    rng = np.random.RandomState(1234)
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class LSTM():
    def __init__(self,n_in,n_hidden,x=None,prefix=""):
         
        self.params = []
        self.x = x
        #if x:
        #    self.x = x
        #else:
        #    self.x = T.matrix("x")

        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        #self.last_hidden = h[-1]
        self.all_hidden = h
        self.nn_out = h[-1]

    def lstm_recurrent_fn(self,x,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf)

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi)

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo)

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc)

        c_t = ft*c_t_1 + it*ct_

        h_t = ot*tanh(c_t)
        return h_t,c_t

class LSTM_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))


        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)
        c_t_0 = T.alloc(0., x.shape[0], n_hidden)

        #h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        #c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def lstm_recurrent_fn(self,x,mask,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf)

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi)

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo)

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc)

        c_t_this = ft*c_t_1 + it*ct_

        h_t_this = ot*tanh(c_t_this)

        c_t = mask[:, None] * c_t_this + (1. - mask)[:, None] * c_t_1
        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t,c_t

class GRU():
    def __init__(self,n_in,n_hidden,x=None,prefix=""):
         
        self.params = []
        self.x = x

        wz_x,bz = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wz_x,bz]

        wr_x,br = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wr_x,br]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]


        wz_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wz_h]     

        wr_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wr_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     


        h_t_1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h,r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_1],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        self.all_hidden = h
        self.nn_out = h[-1]

    def recurrent_fn(self,x,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t = (1-fz)*h_t_1 + fz*h_new

        return h_t


class GRU_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))


        wz_x,bz = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wz_x,bz]

        wr_x,br = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wr_x,br]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]


        wz_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wz_h]     

        wr_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wr_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     


        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)

        #h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        #c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h,r = theano.scan(self.recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def recurrent_fn(self,x,mask,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t_this = (1-fz)*h_t_1 + fz*h_new

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t

class sub_GRU_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),xc=T.tensor3("xc"),mask=T.matrix("mask"),maskc=T.matrix("maskx"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")
        if xc is not None:
            self.xc = xc
        else:
            self.xc = T.tensor3("xc")


        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")
        if maskc is not None:
            self.maskc = maskc
        else:
            self.maskc = T.matrix("maskc")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))

        nmaskc = T.transpose(self.maskc,axes=(1,0))
        nxc = T.transpose(self.xc,axes=(1,0,2))


        wz_x,bz = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wz_x,bz]

        wr_x,br = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wr_x,br]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]


        wz_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wz_h]     

        wr_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wr_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     


        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)
        h_t_0_c = T.alloc(0., xc.shape[0], n_hidden)

        #h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        #c_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        h,r = theano.scan(self.recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        hc,rc = theano.scan(self.recurrent_fn, sequences = [nxc,nmaskc],
                       outputs_info = [h_t_0_c],
                       non_sequences = [wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc])

        self.all_hiddenx = T.transpose(h,axes=(1,0,2))
        self.nn_outx = h[-1]

        self.all_hiddenc = T.transpose(hc,axes=(1,0,2))
        self.nn_outc = hc[-1]

        self.nn_out = h[-1] - hc[-1]

    def recurrent_fn(self,x,mask,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t_this = (1-fz)*h_t_1 + fz*h_new

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t

class sub_LSTM_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),xc=T.tensor3("xc"),mask=T.matrix("mask"),maskc=T.matrix("maskx"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")
        if xc is not None:
            self.xc = xc
        else:
            self.xc = T.tensor3("xc")


        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")
        if maskc is not None:
            self.maskc = maskc
        else:
            self.maskc = T.matrix("maskc")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))

        nmaskc = T.transpose(self.maskc,axes=(1,0))
        nxc = T.transpose(self.xc,axes=(1,0,2))


        wf_x,bf = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [wf_x,bf]

        wi_x,bi = init_weight(n_in,n_hidden,pre="%s_lstm_i_x_"%prefix) 
        self.params += [wi_x,bi]

        wc_x,bc = init_weight(n_in,n_hidden,pre="%s_lstm_c_x_"%prefix) 
        self.params += [wc_x,bc]

        wo_x,bo = init_weight(n_in,n_hidden,pre="%s_lstm_o_x_"%prefix) 
        self.params += [wo_x,bo]


        wf_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [wf_h]     

        wi_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_i_h_"%prefix)
        self.params += [wi_h]     

        wc_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_c_h_"%prefix)
        self.params += [wc_h]     

        wo_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_o_h_"%prefix)
        self.params += [wo_h]     

        #h_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        #c_t_0 = T.alloc(np.array(0.,dtype=np.float64), x.shape[0], n_hidden)
        h_t_0 = T.alloc(0., x.shape[0], n_hidden)
        c_t_0 = T.alloc(0., x.shape[0], n_hidden)

        h_t_0_c = T.alloc(0., xc.shape[0], n_hidden)
        c_t_0_c = T.alloc(0., xc.shape[0], n_hidden)


        [h,c],r = theano.scan(self.lstm_recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0,c_t_0],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        [hc,cc],rc = theano.scan(self.lstm_recurrent_fn, sequences = [nxc,nmaskc],
                       outputs_info = [h_t_0_c,c_t_0_c],
                       non_sequences = [wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo])

        self.all_hiddenx = T.transpose(h,axes=(1,0,2))
        self.nn_outx = h[-1]

        self.all_hiddenc = T.transpose(hc,axes=(1,0,2))
        self.nn_outc = hc[-1]

        self.nn_out = h[-1] - hc[-1]

    def lstm_recurrent_fn(self,x,mask,h_t_1,c_t_1,wf_x,wf_h,bf,wi_x,wi_h,bi,wc_h,wc_x,bc,wo_x,wo_h,bo):
        ft = sigmoid(T.dot(h_t_1,wf_h) + T.dot(x,wf_x) + bf) 

        it = sigmoid(T.dot(h_t_1,wi_h) + T.dot(x,wi_x) + bi) 

        ot = sigmoid(T.dot(h_t_1,wo_h) + T.dot(x,wo_x) + bo) 

        ct_ = tanh(T.dot(h_t_1,wc_h) + T.dot(x,wc_x) + bc) 

        c_t_this = ft*c_t_1 + it*ct_

        h_t_this = ot*tanh(c_t_this)

        c_t = mask[:, None] * c_t_this + (1. - mask)[:, None] * c_t_1
        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t,c_t

    def recurrent_fn(self,x,mask,h_t_1,wz_x,wz_h,bz,wr_x,wr_h,br,wc_x,wc_h,bc):
        fz = sigmoid(T.dot(h_t_1,wz_h) + T.dot(x,wz_x) + bz)

        fr = sigmoid(T.dot(h_t_1,wr_h) + T.dot(x,wr_x) + br)

        h_new = tanh(T.dot(x,wc_x) + T.dot( (fr*h_t_1) ,wc_h) + bc)

        h_t_this = (1-fz)*h_t_1 + fz*h_new

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t

class RNN_batch():
    def __init__(self,n_in,n_hidden,x=T.tensor3("x"),mask=T.matrix("mask"),prefix=""):
         
        self.params = []
        if x is not None:
            self.x = x
        else:
            self.x = T.tensor3("x")

        if mask is not None:
            self.mask = mask
        else:
            self.mask = T.matrix("mask")

        #### 转置 为了进行scan运算 ###
    
        nmask = T.transpose(self.mask,axes=(1,0))
        nx = T.transpose(self.x,axes=(1,0,2))

        w_x,b = init_weight(n_in,n_hidden,pre="%s_lstm_f_x_"%prefix) 
        self.params += [w_x,b]

        w_h,b_h = init_weight(n_hidden,n_hidden,pre="%s_lstm_f_h_"%prefix)
        self.params += [w_h]     

        h_t_0 = T.alloc(0., x.shape[0], n_hidden)

        h,r = theano.scan(self.recurrent_fn, sequences = [nx,nmask],
                       outputs_info = [h_t_0],
                       non_sequences = [w_x,w_h,b])

        self.all_hidden = T.transpose(h,axes=(1,0,2))
        self.nn_out = h[-1]

    def recurrent_fn(self,x,mask,h_t_1,w_x,w_h,b):

        h_t_this = tanh(T.dot(x,w_x) + T.dot(h_t_1,w_h) + b)

        h_t = mask[:, None] * h_t_this + (1. - mask)[:, None] * h_t_1

        return h_t

class RNN():
    def __init__(self,n_in,n_hidden,x=None):
         
        self.params = []
        if x:
            self.x = x
        else:
            self.x = T.matrix("x")

        w_in,b_in = init_weight(n_in,n_hidden,pre="rnn_x") 
        self.params += [w_in]

        w_h,b_h = init_weight(n_hidden,n_hidden,pre="rnn_h")
        self.params += [w_h,b_h]     

        h_t_0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        h, r = theano.scan(self.recurrent_fn, sequences = self.x,
                       outputs_info = [h_t_0],
                       non_sequences = [w_in ,w_h, b_h])

        self.nn_out = h[-1]
        self.all_hidden = h 


    def recurrent_fn(self,x,h_t_1,w_in,w_h,b):
        h_t = sigmoid(T.dot(h_t_1, w_h) + T.dot(x, w_in) + b)
        return h_t

def init_weight_file(fn,dimention=100,pre="embedding"):
    f = open(fn)
    numnum = 1 
    oo = []
    oo.append([0.0]*dimention)
    while True:
        line = f.readline()
        if not line:break
        line = line.strip().split(" ")[1:]
        numnum += 1
        if numnum%100000 == 0:print >> sys.stderr,numnum
        out = [float(t.strip()) for t in line]
        if not len(out) == dimention:continue
        oo.append(out)
    #print oo
    W_values = np.asarray(oo,dtype = theano.config.floatX)
    #print oo
    #W_values = np.array(oo)
    #print W_values.shape
    w = theano.shared(
        value=W_values,
        name='%sw'%pre, borrow=True
    )   
    return w
