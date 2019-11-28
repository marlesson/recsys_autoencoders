from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.layers import add
from model.BaseModel import BaseModel
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AutoEncContentModel(BaseModel):
  '''
  Model adapted from Auto Encoder With Content Base Information

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

    # Params
    #   integer encode the documents
    vocab_size   = 100
    #   pad documents to a max length of 4 words
    max_length   = 50


    def split_str(val):
      '''
      Split and Join Array(Array(str))
      '''
      tokens = []
      for v in val:
          tokens.extend(v.split(' '))
      return ' '.join(tokens)

    #  Order users in matrix interactions
    users_ids  = list(user_item_matrix.index)
    
    # Dataset with User X Content information
    user_games = interactions.groupby('user_id')['game'].apply(list).loc[users_ids].reset_index()
    user_games['tokens'] = user_games['game'].apply(split_str)

    # Prepare input layer
    encoded_tokens = [one_hot(d, vocab_size) for d in user_games.tokens]
    padded_tokens  = pad_sequences(encoded_tokens, maxlen=max_length, padding='post')

    # Input  
    X = [user_item_matrix.values, padded_tokens]
    y = user_item_matrix.values

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

    # Params
    users_items_matrix, content_info = X

    # Input
    input_layer   = x = Input(shape=(users_items_matrix.shape[1],), name='UserScore')
    input_content = Input(shape=(content_info.shape[1],), name='Itemcontent')

    # Encoder
    k = int(len(self.layers)/2)
    i = 0
    for l in self.layers[:k]:
      x = Dense(l, activation=self.activation, 
                      name='EncLayer{}'.format(i))(x)
      i = i+1

    # Latent Space
    x = Dense(self.layers[k], activation=self.activation, 
                                name='UserLatentSpace')(x)

    # Content Information
    x_content = Embedding(100, self.layers[k], 
                        input_length=content_info.shape[1])(input_content)
    x_content = Flatten()(x_content)
    x_content = Dense(self.layers[k], activation=self.activation, 
                                name='ItemLatentSpace')(x_content)
    # Concatenate
    x = add([x, x_content], name='LatentSpace')

    # Dropout
    x = Dropout(self.dropout)(x)

    # Decoder
    for l in self.layers[k+1:]:
      i = i-1
      x = Dense(l, activation=self.activation, 
                      name='DecLayer{}'.format(i))(x)

    # Output
    output_layer = Dense(users_items_matrix.shape[1], activation='linear', name='UserScorePred')(x)


    # this model maps an input to its reconstruction
    model = Model([input_layer, input_content], output_layer)

    return model
