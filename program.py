# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:38:47 2021

@author: FaSa79_
"""

import cv2
import numpy as np
import os
import time

new_frame_time = 0
prev_frame_time = 0

os.chdir('D:/OneDrive - International Islamic University Malaysia/Studies/3rd Year 2nd Sem/CSC4903 Final Year Project 1/Exercise/Yolov3')

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
classes = []

with open('coco.names', 'r') as f:
    classes =  f.read().splitlines()
    
cap = cv2.VideoCapture(0)

while True:
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    
    _, img = cap.read()
    height, width, _ = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)    
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()
