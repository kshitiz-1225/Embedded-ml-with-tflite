# -*- coding: utf-8 -*-
# Import the necessary Libraries

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras

import cv2
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import random

"""Import the MNIST Dataset"""

mnist = keras.datasets.mnist

load_data = mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = load_data

# normalization factor as 255.0
normalize_factor = 255.0

print(train_images)

print(train_labels)

print(test_images, test_labels)

print(len(train_images))

print(len(train_labels))

print(len(test_images), len(test_labels))

train_images = train_images / normalize_factor
test_images = test_images / normalize_factor

"""Visualize the Dataset"""

plt.figure(figsize=(8,8))
for i in range(16):
  plt.xticks([])
  plt.subplot(4,4,i+1)
  plt.xlabel(train_labels[i])
  plt.grid(False)
  plt.imshow(train_images[i])
  plt.yticks([])

plt.show()

plt.figure(figsize=(8,8))
for i in range(16):
  plt.xticks([])
  plt.subplot(4,4,i+1)
  plt.xlabel(train_labels[i])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.gray)
  plt.yticks([])

plt.show()

def images_and_pixels(image, ax):

    '''Displays a random image along with its pixel values'''

    ax.imshow(image)
    width, height = image.shape
    threshold = image.max() / 2.5

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(image[x][y],2)),
                        xy = (y,x),
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        color = 'white' if image[x][y] < threshold else 'black')

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111)
random_number = random.randint(0, 10)

images_and_pixels(train_images[random_number], ax)

# plt.title('Label = ' + str(test_images[random_number]), fontsize = 20)
plt.axis('off')
plt.show()

g = sns.countplot(test_labels)
print(g)

"""Model Architecture"""

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation=tf.nn.relu),
  keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Dropout(0.25),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size = (2,2), strides =(2,2)),
  keras.layers.Dropout(0.25),
  keras.layers.Flatten(),
  keras.layers.Dense(256),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10)
])

# Define how to train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the digit classification model
model.fit(train_images, train_labels, epochs=5)

print(model.summary())

tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)

"""Test Accuracy"""

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# if input parameter matches --> Blue else Yellow
def get_label_color(val1, val2):
  if val1 == val2:
    return 'blue'
  else:
    return 'yellow'

# Predict the digit label
predictions = model.predict(test_images)

prediction_digits = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 12))
for i in range(36):
  ax = plt.subplot(6, 6, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  image_index = random.randint(0, len(prediction_digits))
  plt.imshow(test_images[image_index], cmap=plt.cm.gray)
  ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index],\
                                           test_labels[image_index]))
  plt.xlabel('Predicted: %d' % prediction_digits[image_index])
plt.show()

"""Convert to TensorflowLite Model"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()
import warnings
warnings.filterwarnings('ignore')

# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)

print('Quantized model is about %d%% of the float model size.' % (100 * quantized_model_size / float_model_size))

"""Evaluation of the Tensorflow-Lite Model"""

def evaluate_tflite_model(tflite_model):
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  prediction_digits = []
  input_tensor_index = interpreter.get_input_details()[0]["index"]
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
  for test_image in test_images:
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image)

    interpreter.invoke()

    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

"""Evaluate the TFLite Float Model (Model size is large as compared to quantized model)"""

float_accuracy = evaluate_tflite_model(tflite_float_model)
print('Float model accuracy = %.4f' % float_accuracy)

"""Evaluate the TFLite Quantized Model (Model size is made small for easier deployment)"""

quantized_accuracy = evaluate_tflite_model(tflite_quantized_model)
print('Quantized model accuracy = %.4f' % quantized_accuracy)

print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))

"""Download the TF_Float file"""

from google.colab import files
f = open('mnist_float.tflite', "wb")
f.write(tflite_float_model)
f.close()
files.download('mnist_float.tflite')
print("`mnist_float.tflite` has been downloaded")

"""Download the TF_Quantized File"""

from google.colab import files
f = open('mnist_quantized.tflite', "wb")
f.write(tflite_quantized_model)
f.close()
files.download('mnist_quantized.tflite')
print("`mnist_quantized.tflite` has been downloaded")