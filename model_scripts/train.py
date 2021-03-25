#fitting images to CNN
history = classifier.fit_generator(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=50,
                         validation_steps=valid_num//batch_size,
                         callbacks=callback_list)
