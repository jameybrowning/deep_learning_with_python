# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:41:37 2019


"""
#load model
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

#Load and display a cat image
img_path = r'C:\Users\VAMS_2\Dropbox\ML\Deep_Learning_With_Python\Dogs_vs_cats\Dogs_vs_cats_small\test\cats\cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size = (150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)

#display image
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

# create a multi-output layer activations model and run in predict mode
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

activations = activation_model.predict(img_tensor)


#visualize tenth channel of first layer
plt.matshow(activations[0][0,:,:,10], cmap = 'viridis')

#Visualize every channel of every layer

    
# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()

# visualizing convnet filters

#define loss tensor for filter visualization

    
            
            
    
