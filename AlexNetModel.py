#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
os.listdir("./input/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)")


# **Building CNN Based On AlexNet Architecture**

# In[3]:


# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

# Convolution Step 2
classifier.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

# Flattening Step
classifier.add(Flatten())

# Full Connection Step 1
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))

# Full Connection Step 2
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))

# Classification step 
classifier.add(Dense(units = 38, activation = 'softmax'))

# Model summary
classifier.summary()


# **Compiling the Model**

# In[4]:


# Compiling the Model
from keras import optimizers
classifier.compile(optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# **Image Preprocessing**

# In[8]:


# image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128

base_dir = "./input/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

training_set = train_datagen.flow_from_directory(base_dir+'/train',
                                                 target_size=(227, 227),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',
                                            target_size=(227, 227),
                                            batch_size=batch_size,
                                            class_mode='categorical')

train_num = training_set.samples
valid_num = valid_set.samples


# **Checkpoints**

# In[10]:


# checkpoints
from keras.callbacks import ModelCheckpoint
weightpath = "AlexNet_Weights.h5"
checkpoints = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callback_list = [checkpoints]


# **Model Training**

# In[11]:


#fitting images to CNN
history = classifier.fit_generator(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=50,
                         validation_steps=valid_num//batch_size,
                         callbacks=callback_list)


# **Model Saving**

# In[ ]:


#saving the trained model
filepath="AlexNetModel.h5"
classifier.save(filepath)


# **Visualising Training Progress**

# In[ ]:


#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

