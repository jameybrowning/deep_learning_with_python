# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:45:56 2019

@author: James Browning
"""

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data[0]
train_labels[0]
max([max(sequence) for sequence in train_data])

word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


#vectorize training data[0,1]
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i,sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data) 

#vectorize label data
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#define the model

from keras import models
from keras import layers
from keras import regularizers

model1 = models.Sequential()
model1.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model1.add(layers.Dense(16, activation = 'relu'))
model1.add(layers.Dense(1, activation = 'sigmoid'))

model2 = models.Sequential()
model2.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                       activation = 'relu', input_shape = (10000,)))
model2.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                       activation = 'relu'))
model2.add(layers.Dense(1, activation = 'sigmoid'))


#compile model with defualt parameters
"""

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
"""

#compile model, configure optimizer, losses, and metrics
from keras import optimizers
from keras import losses
from keras import metrics

#Set aside validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Compile and train model
model1.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history1 = model1.fit(partial_x_train, partial_y_train, 
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

model2.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history2 = model2.fit(partial_x_train, partial_y_train, 
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))




#plot validation loss for base model and l2 model
import matplotlib.pyplot as plt
history_dict = history1.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, val_loss_values, 'b', label='Validation loss Base')
history_dict = history2.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, val_loss_values, 'bo', label='Validation loss L2')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
   
#plot training and validation accuracy

"""
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()                            
                               
model.predict(x_test)

"""








