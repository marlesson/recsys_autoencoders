from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import initializers
from keras.layers import add
from model.BaseModel import BaseModel
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

  def __init__(self, layers, epochs, batch, activation, dropout, lr, reg):
    self.layers  = eval(layers)
    self.epochs  = epochs
    self.batch   = batch
    self.activation = activation
    self.dropout = dropout
    self.lr      = lr
    self.reg     = reg
    self.model   = None

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
    '''
    Autoencoder for Collaborative Filter Model
    '''

    # Input
    input_layer = x = Input(shape=(X.shape[1],))
    #x = Dropout(0.6)(x)
    # "encoded" is the encoded representation of the input
    k = int(len(self.layers)/2)
    i = 0
    for l in self.layers[:k]:
      x = Dense(l, activation=self.activation, 
                      name='enc_{}'.format(i))(x)
      i = i+1

    # Latent Space
    x = Dense(self.layers[k], activation=self.activation, 
                                name='latent_space')(x)
    # Dropout
    x = Dropout(self.dropout)(x)

    for l in self.layers[k+1:]:
      i = i-1
      x = Dense(l, activation=self.activation, 
                      name='dec_{}'.format(i))(x)

    # Output
    output_layer = Dense(X.shape[1], activation='linear', name='output')(x)


    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)

    return model
