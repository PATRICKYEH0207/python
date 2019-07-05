from __future__ import absolute_import, division, print_function
import os
from os import listdir
from os.path import isfile, isdir, join
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version is ", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from img_train_test_split import *
import time 
tStart = time.time()#stra time
epochs = 10
#Create Image Data Generator with Image Augmentation
image_size = 160 # All images will be resized to 160x160
batch_size = 32
#file_writer = tf.summary.FileWriter("D:\python\logs")
#Load data
#path =r'.\cats_and_dogs_filtered'
#splitfile=img_train_test_split(path,0.8)
#base_dir, _ = os.path.splitext(".\data")#重點:路徑明成最前面要加r
#print(base_dir)
#print( _ )
base_dir=r'.\cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
print(train_dir)
print(validation_dir)

# Directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'Normal')
print ('Total training dogs images:', len(os.listdir(train_dogs_dir)))

# Directory with our training dog pictures
train_cats_dir = os.path.join(train_dir, 'Stroke')
print ('Total training cats images:', len(os.listdir(train_cats_dir)))

# Directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'Normal')
print ('Total validation dogs images:', len(os.listdir(validation_dogs_dir)))

# Directory with our validation dog pictures
validation_cats_dir = os.path.join(validation_dir, 'Stroke')
print ('Total validation cats images:', len(os.listdir(validation_cats_dir)))



# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),  
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='binary')

#Create the base model from the pre-trained convnets
IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
#讀取NET(MobileNet V2)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')

#Freeze the convolutional base
base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()

#Add a classification head
model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(1, activation='sigmoid')
])

#Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 
              loss='binary_crossentropy', 
              metrics=['acc'])
model.summary()#output model
len(model.trainable_variables)

#Train the model

steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs, 
                              workers=4,
                              validation_data=validation_generator, 
                              validation_steps=validation_steps)

tEnd = time.time()#end_time
print ("It cost %f sec" % (tEnd - tStart))#full time

#Learning curves
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()


#file_writer.add_graph(sess.graph)
print ('-------------finsh-------------')
