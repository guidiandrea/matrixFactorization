
import mxnet as mx 
from mxnet import gluon, autograd, nd 
import numpy as np

class NCF(gluon.Block):
    
    def __init__(self,user_vocabulary,user_emb_size,item_vocabulary,item_emb_size,hidden_units,n_outputs,**kwargs):
        
        super(NCF,self).__init__(**kwargs)
        
        with self.name_scope():
            
            self.user_emb = gluon.nn.Embedding(input_dim=user_vocabulary+1,
                                               output_dim=user_emb_size)
            self.item_emb = gluon.nn.Embedding(input_dim=item_vocabulary+1,
                                               output_dim=item_emb_size)
            hidden_units = hidden_units
            n_layers = len(hidden_units)
                
            for i in range(n_layers):
                
                self.__dict__[f'dense_{i+1}'] = gluon.nn.Dense(units=hidden_units[i], activation='relu')
                self.__dict__['_children'][f'dense_{i+1}'] = self.__dict__[f'dense_{i+1}']
        
            self.out_layer = gluon.nn.Dense(units=n_outputs)
                    
    def forward(self, x,y):
        
        x = self.user_emb(x)
        y = self.item_emb(y)
        latent_vector = nd.elemwise_mul(x,y)
        hid = self.dense_1(latent_vector)
        hid = self.dense_2(hid)
        hid = self.dense_3(hid)
        out = self.out_layer(hid)
        
        return(out)
                                  
########################################################################################

class InputDataset(gluon.data.Dataset):
    
    def __init__(self,data, columns=None):
    
        self.length = len(data)
        # Supposing that if it's not numpy or ndarray then it's a dataframe
        if not isinstance(data,mx.ndarray.ndarray.NDArray):
            self.data = _dataframe_to_ndarray(data,columns)
        self.data = data
        
    def __getitem__(self, idx):
        
        userId = self.data[idx,0].astype('int32')
        itemId = self.data[idx,1].astype('int32')
        rating = self.data[idx,2]
        
        return userId,itemId,rating
    
    def __len__(self):
        
        return self.length
    
    
    def _dataframe_to_ndarray(data, columns):
        
        return nd.array(data[columns])
    
        