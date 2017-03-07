#coding=utf8
from NetworkComponet import *


class Net():
    def __init__(self,embedding_dimention,hidden_layer_dimention,voc_size,batch_size,window_size=4):

        self.params = []
        # generate embedding in random
        self.w_embedding,b_zp = init_weight(voc_size,embedding_dimention)
        # generate embedding by file
        #self.w_embedding = init_weight_file(args.embedding_file,args.embedding_dimention)
        
        self.x = T.imatrix("X")
        self.w_embedding_sub = self.w_embedding[self.x.flatten()]
        
        self.inpt = self.w_embedding_sub
        self.inpt = self.inpt.reshape((batch_size,window_size*embedding_dimention))
        
        linearLayer = Layer(window_size * embedding_dimention,hidden_layer_dimention,self.inpt,linear)
        self.params += linearLayer.params

        vocOutputLayer = Layer(hidden_layer_dimention,voc_size,linearLayer.output,linear)
        self.params += vocOutputLayer.params

        coreOutputLayer = Layer(hidden_layer_dimention,10,linearLayer.output,linear)
        self.params += coreOutputLayer.params

        vocResult = softmax(vocOutputLayer.output)
        coreResult = softmax(coreOutputLayer.output)

        self.get_voc = theano.function(inputs=[self.x],outputs=[vocResult])
        self.get_coref = theano.function(inputs=[self.x],outputs=[coreResult])

        #l1_norm_squared = sum([(w**2).sum() for w in self.params])
        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])

        #lmbda_l1 = 0.0 
        lmbda_l2 = 0.001 

        y1 = T.ivector('y1')
        y2 = T.ivector('y2')
        y3 = T.ivector('y3')
        y = T.ivector('voc')

        ###y对应着 第一种分类 y=[0,2,1] 表示 batch=3的情况下，每个case的分类分别为0 2 1
        cost = T.mean(-(T.log(vocResult)[T.arange(y.shape[0]), y])) +\
            T.mean(-(T.log(coreResult)[T.arange(y1.shape[0]), y1])) +\
            T.mean(-(T.log(coreResult)[T.arange(y2.shape[0]), y2])) +\
            T.mean(-(T.log(coreResult)[T.arange(y3.shape[0]), y3])) +\
            lmbda_l2*l2_norm_squared

        lr = T.scalar()

        sub_grad = T.grad(cost, self.w_embedding_sub)
        sub_updates = T.set_subtensor(self.w_embedding_sub, self.w_embedding_sub-lr*sub_grad)

        para_updates_all = lasagne.updates.sgd(cost, self.params, lr)
        para_updates = [(p,para_updates_all[p]) for p in para_updates_all] 


        self.train_step = theano.function(inputs=[self.x,y,y1,y2,y3,lr], outputs=[cost],
            on_unused_input='warn',
            updates=[(self.w_embedding,sub_updates)]+para_updates
            #updates=updates
            #allow_input_downcast=True
            )

    def show_para(self):
        return self.w_embedding.get_value()



def main():

    net = Net(3,10,5,2)
    x = [[1,2,3,4]]*2

    print net.get_voc(x)
    print net.get_coref(x)
    print net.show_para()

    net.train_step(x,[1,0],[0,2],[3,4],[5,9],2)
    net.train_step(x,[1,0],[0,2],[3,4],[5,9],2)
    net.train_step(x,[1,0],[0,2],[3,4],[5,9],2)
    
    print net.get_voc(x)
    print net.get_coref(x)

    print net.show_para()
if __name__ == "__main__":
    main()

