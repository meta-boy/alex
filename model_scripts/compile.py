# Compiling the Model

from keras import optimizers
classifier.compile(optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])