#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:31:52 2018

@author: fadlimuharram
"""
from flask import Flask
from flask_restful import Resource, Api

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd



webapp = Flask(__name__)
api = Api(webapp)


class Quotes(Resource):
    
    def __init__(self):
        # initialising the cnn
        self.klasifikasi = Sequential()
        
        # step 1 - convolution
        self.klasifikasi.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        print("Convolution of layer 1 Completed")
        
        
        # step 2 - pooling
        self.klasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
        print("Pooling Completed")
        
        # adding a second convolution layer
        self.klasifikasi.add(Convolution2D(32, 3, 3, activation='relu'))
        self.klasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
        
        
        
        # step 3 - flattening
        self.klasifikasi.add(Flatten())
        print("Flattening Completed")
        
        # step 4 -full connection
        self.klasifikasi.add(Dense(units = 128, activation = 'relu'))
        self.klasifikasi.add(Dense(output_dim = 128, activation='relu'))
        self.klasifikasi.add(Dense(units = 3, activation = 'softmax'))
        print("Full Connection Between Hidden Layers and Output Layers Completed")
        
        # kompile cnn
        self.klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        print("Compiling Initiated")
        
        
        # part 2 -fitting the cnn to images
        
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_set = train_datagen.flow_from_directory(
                'anggrek/training_set',
                target_size=(64, 64),
                batch_size=8,
                class_mode='categorical')
        
        self.test_set = test_datagen.flow_from_directory(
                'anggrek/testing_set',
                target_size=(64, 64),
                batch_size=8,
                class_mode='categorical')
        
        self.datanya = self.klasifikasi.fit_generator(
                self.train_set,
                steps_per_epoch=43,
                epochs=5,
                validation_data=self.test_set,
                validation_steps=26)
        print("Compiling Completed")
    
    def get(self):
        test_image = image.load_img('F3.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.klasifikasi.predict(test_image).tolist()
        result = pd.Series(result).to_json(orient='values')
        self.train_set.class_indices

        return {
            'ataturk': {
                'quote': ['Yurtta sulh, cihanda sulh.', 
                    'Egemenlik verilmez, alınır.', 
                    'Hayatta en hakiki mürşit ilimdir.']
            },
            'linus': {
                'quote': ['Talk is cheap. Show me the code.']
            },
            'hasil': result

        }


api.add_resource(Quotes, '/')

if __name__ == '__main__':
    webapp.run(debug=True)
    
