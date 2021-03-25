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