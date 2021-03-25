
# checkpoints
from keras.callbacks import ModelCheckpoint
weightpath = "AlexNet_Weights.h5"
checkpoints = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callback_list = [checkpoints]