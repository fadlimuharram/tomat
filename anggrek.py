#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 05:19:56 2018

@author: fadlimuharram
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# initialising the cnn
klasifikasi = Sequential()

# step 1 - convolution
klasifikasi.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
print("Convolution of layer 1 Completed")


# step 2 - pooling
klasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
print("Pooling Completed")

# adding a second convolution layer
klasifikasi.add(Convolution2D(32, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D(pool_size=(2, 2)))



# step 3 - flattening
klasifikasi.add(Flatten())
print("Flattening Completed")

# step 4 -full connection
klasifikasi.add(Dense(units = 128, activation = 'relu'))
klasifikasi.add(Dense(output_dim = 128, activation='relu'))
klasifikasi.add(Dense(units = 3, activation = 'softmax'))
print("Full Connection Between Hidden Layers and Output Layers Completed")





# kompile cnn
klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
print("Compiling Initiated")


# part 2 -fitting the cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'anggrek/training_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'anggrek/testing_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

datanya = klasifikasi.fit_generator(
        train_set,
        steps_per_epoch=43,
        epochs=10,
        validation_data=test_set,
        validation_steps=26)
print("Compiling Completed")



# membuat prediksi
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('bc.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = klasifikasi.predict(test_image)
train_set.class_indices

print(datanya.history.keys())
print(datanya.history['acc'])
print(datanya.history['val_acc'])

import matplotlib.pyplot as plt

plt.plot(datanya.history['acc'])
plt.plot(datanya.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# history untuk loss
plt.plot(datanya.history['loss'])
plt.plot(datanya.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# simpan ke json
from keras.models import model_from_json

# serialize model ke JSON
model_json = klasifikasi.to_json()
with open("image_classifier.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
klasifikasi.save_weights("image_classifier.h5")
print("Saved model to disk")

# evaluate loaded model on test data
json_file = open('image_classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_model_json)
# load weights into new model
loaded_classifier.load_weights("image_classifier.h5")
print("Loaded model from disk")

#test_image_json = image.load_img('hc.jpg', target_size = (64, 64))
#test_image_json = image.img_to_array(test_image_json)
#test_image_json = np.expand_dims(test_image_json, axis = 0)
#result_json = loaded_classifier.predict(test_image_json)

#train_set.class_indices
#list(train_set.class_indices.keys())
#result_json[0]
#list_prediction_classes = [list(train_set.class_indices.keys())[i] for i in result_json] 

# --- check accuracy dari load model
loaded_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores_2 = loaded_classifier.evaluate_generator(train_set, steps=4661)
print("%s: %.2f, %s: %.2f%%" % (loaded_classifier.metrics_names[0], scores_2[0], loaded_classifier.metrics_names[1], scores_2[1]*100))



from skimage.io import imread
from skimage.transform import resize
img = imread('pn.jpg') 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
 
if(np.max(img)>1):
    img = img/255.0
    
prediction = loaded_classifier.predict_classes(img)

list_prediction_classes = [list(train_set.class_indices.keys())[i] for i in prediction]

list_prediction_classes

