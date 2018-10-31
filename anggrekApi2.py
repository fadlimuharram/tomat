#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:22:00 2018

@author: fadlimuharram
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from keras.preprocessing import image
import flask

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
klasifikasi = None
train_set = None
test_set = None
datanya = None

def load_model():
    global klasifikasi, train_set, test_set, datanya
    # initialising the cnn
    klasifikasi = Sequential()
    
    # step 1 - convolution
    klasifikasi.add(Convolution2D(32, # jumlah filter layer
                                    3, # y dimensi untuk kernel
                                    3, # x dimensi untuk kernel
                                    input_shape=(64, 64, 3), 
                                    activation='relu'))
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
    
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(
            'tomat/color/sample/training_set',
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            'tomat/color/sample/testing_set',
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    datanya = klasifikasi.fit_generator(
        train_set,
        steps_per_epoch=43,
        epochs=5,
        validation_data=test_set,
        validation_steps=26)
    
    klasifikasi._make_predict_function()
    print("Compiling Completed")

@app.route("/predict")
def predictnya():
    global train_set, klasifikasi
    
    #data = {"success": False}
    
    test_image = image.load_img('F3.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = klasifikasi.predict(test_image).tolist()
    result = pd.Series(result).to_json(orient='values')
    #train_set.class_indices
    
    #preds = klasifikasi.predict(test_image)
    #print(preds)
    #results = imagenet_utils.decode_predictions(preds)
    #data["predictions"] = []
    
    #for (imagenetID, label, prob) in results[0]:
    #    r = {"label": label, "probability": float(prob)}
    #    data["predictions"].append(r)

    # indicate that the request was a success
    #data["success"] = True
    
    print(result)
        
    return flask.jsonify(result)

        

  
        
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(debug=True)