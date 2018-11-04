#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:40:28 2018

@author: fadlimuharram
"""
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# evaluate loaded model on test data
json_file = open('hasil/tomatVgg_classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_model_json)
# load weights into new model
loaded_classifier.load_weights("hasil/tomatVgg_classifier.h5")
print("Loaded model from disk")
loaded_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])

test_image = image.load_img('b913f1de-86ed-447d-bb74-9310460de8a9___GCREC_Bact.Sp 3149_final_masked.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

prediction = loaded_classifier.predict_classes(test_image)
print(prediction)
