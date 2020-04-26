
import mxnet as mx 
from mxnet import gluon, autograd, nd 
import numpy as np

class NCF(gluon.Block):
    
    
    def __init__(self,user_vocabulary,user_emb_size,item_vocabulary,item_emb_size,n_hidden_layers,hidden_units,n_outputs,**kwargs):
        
        super(NCF,self).__init__(**kwargs)
        
        with self.name_scope():
            
            self.user_emb = gluon.nn.Embedding(input_dim=user_vocabulary+1,
                                               output_dim=user_emb_size)
            self.item_emb = gluon.nn.Embedding(input_dim=item_vocabulary+1,
                                               output_dim=item_emb_size)

            n_hidden_layers = n_hidden_layers
            hidden_units = hidden_units
            
            if n_hidden_layers != len(hidden_units):
                
                raise ValueError("You have to specify as many hidden units as layers")
                
            #for i in range(n_hidden_layers):
                
            #    self.__dict__[f'inner_dense_{i+1}'] = gluon.nn.Dense(units=hidden_units[i], activation='relu')
        
            self.dense_1 = gluon.nn.Dense(units=hidden_units[0], activation='relu')
            self.dense_2 = gluon.nn.Dense(units=hidden_units[1], activation='relu')
            self.out_layer = gluon.nn.Dense(units=n_outputs)
                    
    def forward(self, x,y):
        
        x = self.user_emb(x)
        print(f"shape of x:{x}")
        y = self.item_emb(y)
        print(f"shape of y:{y}")
        latent_vector = nd.concat(x,y, dim=1)
        print(f"shape of concatenated:{latent_vector}")

        hid = self.dense_1(latent_vector)
        
        out = self.out_layer(hid)
        
        return(out)
                    
        
                    
                    
class MaskedSumOfSquares(gluon.loss.Loss):
    
    def __init__(self,**kwargs):
        super(MaskedSumOfSquares, self).__init__(weight=None,batch_axis=0,**kwargs)
        
    def forward(self,output,label):
        
        mask = nd.greater(label,0)
        
        masked_output = nd.elemwise_mul(output,mask)
        
        return nd.sum(nd.abs(masked_output-label)**2)         
            
            