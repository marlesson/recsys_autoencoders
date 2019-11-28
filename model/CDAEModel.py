#
#
#
#
# 
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from model.BaseModel import BaseModel
import numpy as np

class CDAEModel(BaseModel):
  '''
  create model
  Reference:
    Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester.
      Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
        The 9th ACM International Conference on Web Search and Data Mining (WSDM'16), p153--162, 2016.  
  '''

  def __init__(self, factors = None, epochs = None, batch = None, 
                      activation = None, dropout = None, lr = None, reg = None):
    self.factors = factors
    self.epochs  = epochs
    self.batch   = batch
    self.activation = activation
    self.dropout = dropout
    self.lr      = lr
    self.reg     = reg
    self.model   = None


  def data_preparation(self, interactions, user_item_matrix):
    '''
    Create a Input to Model
    '''
    users_ids  = list(user_item_matrix.index)
    x_user_ids = np.array(users_ids).reshape(len(users_ids), 1)
    
    X = [user_item_matrix.values, x_user_ids]
    y = user_item_matrix.values

    return X, y

  def fit(self, X, y):
    # Build model
    model = self.build_model(X)

    model.compile(optimizer = Adam(lr=self.lr), 
                    loss='mse')#'mean_absolute_error'

    # train
    hist = model.fit(x=X, y=y,
                      epochs=self.epochs,
                      batch_size=self.batch,
                      shuffle=True,
                      validation_split=0.1,
                      callbacks=self.callbacks_list())

    # Melhor peso
    model.load_weights(self.WEIGHT_MODEL)
    self.model = model

    return model, hist


  def predict(self, X):

    # Predict
    pred = self.model.predict(X)

    # remove watched items from predictions
    pred = pred * (X[0] == 0) 

    return pred

  def build_model(self, X):
    
    # Params
    users_items_matrix, x_user_ids = X

    # Model
    x_item = Input((users_items_matrix.shape[1],), name='UserScore')
    h_item = Dropout(self.dropout)(x_item)
    h_item = Dense(self.factors, 
                      kernel_regularizer=l2(self.reg), 
                      bias_regularizer=l2(self.reg), 
                      activation=self.activation)(h_item)

    # dtype should be int to connect to Embedding layer
    x_user = Input((1,), dtype='int32', name='UserContent')
    h_user = Embedding(len(np.unique(x_user_ids))+1,self.factors, 
                        input_length=1, 
                        embeddings_regularizer=l2(self.reg))(x_user)
    h_user = Flatten()(h_user)

    h = add([h_item, h_user], name='LatentSpace')
    y = Dense(users_items_matrix.shape[1], activation='linear', name='UserScorePred')(h)

    return Model(inputs=[x_item, x_user], outputs=y)
