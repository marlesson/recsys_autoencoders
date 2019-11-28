#
#
#
#
# 

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation
from model.BaseModel import BaseModel
from tensorflow.keras.models import Model

import numpy as np

class AutoEncModel(BaseModel):
  '''
  create model
  Reference:
    KUCHAIEV, Oleksii; GINSBURG, Boris. Training deep autoencoders for collaborative filtering. 
    arXiv preprint arXiv:1708.01715, 2017. 
  https://github.com/NVIDIA/DeepRecommender
  https://arxiv.org/pdf/1708.01715.pdf
  '''

  def __init__(self, layers = '[]', epochs = None, batch = None, 
                      activation = None, dropout = None, lr = None, reg = None):
    self.layers  = eval(layers)
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

    X, y = user_item_matrix.values, user_item_matrix.values

    return X, y


  def fit(self, X, y):
    '''
    Train Model
    '''

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
    '''
    Autoencoder for Collaborative Filter Model
    '''

    # Input
    input_layer = x = Input(shape=(X.shape[1],), name='UserScore')
    
    # Encoder
    # -----------------------------
    k = int(len(self.layers)/2)
    i = 0
    for l in self.layers[:k]:
      x = Dense(l, activation=self.activation, 
                      name='EncLayer{}'.format(i))(x)
      i = i+1

    # Latent Space
    # -----------------------------
    x = Dense(self.layers[k], activation=self.activation, 
                                name='LatentSpace')(x)
    # Dropout
    x = Dropout(self.dropout, name='Dropout')(x)

    # Decoder
    # -----------------------------
    for l in self.layers[k+1:]:
      i = i-1
      x = Dense(l, activation=self.activation, 
                      name='DecLayer{}'.format(i))(x)

    # Output
    output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred')(x)


    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)

    return model
