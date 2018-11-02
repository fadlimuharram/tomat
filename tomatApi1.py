#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:42:06 2018

@author: fadlimuharram
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from keras.preprocessing import image
import flask
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
klasifikasi = None
train_set = None
test_set = None
datanya = None

UPLOAD_FOLDER = '/Users/fadlimuharram/Documents/cnn/upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def load_model():
    global klasifikasi, train_set, test_set, datanya
    # initialising the cnn
    klasifikasi = Sequential()
    
    klasifikasi.add(ZeroPadding2D((1,1),input_shape=(64, 64, 3)))
    klasifikasi.add(Convolution2D(64, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(64, 3, 3, activation='relu'))
    klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(128, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(128, 3, 3, activation='relu'))
    klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
    klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(ZeroPadding2D((1,1)))
    klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
    klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

    klasifikasi.add(Flatten())
    klasifikasi.add(Dense(4096, activation='relu'))
    klasifikasi.add(Dropout(0.5))
    klasifikasi.add(Dense(4096, activation='relu'))
    klasifikasi.add(Dropout(0.5))
    klasifikasi.add(Dense(7, activation='softmax'))
    
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
            'tomat/color/dummy/training_set',
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            'tomat/color/dummy/testing_set',
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    datanya = klasifikasi.fit_generator(
        train_set,
        steps_per_epoch=809,
        epochs=10,
        validation_data=test_set,
        validation_steps=409)
    
    klasifikasi._make_predict_function()
    print("Compiling Completed")
    
    ''' 
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
    klasifikasi.add(Dense(units = 7, activation = 'softmax'))
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
        steps_per_epoch=8860,
        epochs=10,
        validation_data=test_set,
        validation_steps=5428)
    
    klasifikasi._make_predict_function()
    print("Compiling Completed")'''



@app.route("/predict")
def predictnya():
    global train_set, klasifikasi
    
    #data = {"success": False}
    
    test_image = image.load_img('F3.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = klasifikasi.predict(test_image).tolist()
    result = pd.Series(result).to_json(orient='values')
    print(train_set.class_indices)
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

        
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global train_set, klasifikasi
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
    
            test_image = image.load_img('upload/' + filename, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = klasifikasi.predict(test_image).tolist()
            result = pd.Series(result).to_json(orient='values')
            print(train_set.class_indices)
            '''return redirect(url_for('uploaded_file',filename=filename))'''
            print(result)
        
            return flask.jsonify(result)

            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
  
        
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host='192.168.43.70', port=5050,debug=True)