#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:25:48 2018

@author: ankit
"""
img_pred = cv2.imread ('fear1.jpeg', 0)
img_pred.shape
# forces the image to have the input dimensions equal to those used in the training data (28x28)
if img_pred.shape!= [48,48]:
    img2 = cv2.resize (img_pred, (48,48))
    img_pred = img2.reshape ((48,48,1))
else:
    img_pred = img_pred.reshape ((48,48,1))
    

# here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
img_pred = img_pred.reshape (1,48,48,1)

pred = model.predict_classes (img_pred)

pred_proba = model.predict_proba (img_pred)