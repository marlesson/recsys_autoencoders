from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class BaseModel(object):
  WEIGHT_MODEL       = "./artefacts/weights-best-model.hdf5"

  def callbacks_list(self, monitor='val_loss', path_model = WEIGHT_MODEL, patience=15):
    '''
    Callbacks of Train model
    '''
    checkpoint = ModelCheckpoint(path_model, monitor=monitor, verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor=monitor,min_delta=0, patience=patience, verbose=0, mode='auto')

    callbacks_list = [checkpoint, early_stop]
    return callbacks_list    