#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
os.chdir(r"C:\Users\zbook 17 g3\Desktop\cv\dataset")

def resize_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))

cap = cv2.VideoCapture(r"C:\Users\zbook 17 g3\Desktop\cv\cars.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

output_file = 'output_with_motion_vectors.mp4'
frame_width = int(frame1.shape[1] * 0.5)
frame_height = int(frame1.shape[0] * 0.5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height * 2)) 

color = (0, 255, 0)  
thickness = 2 
scale_factor = 0.5  

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = next_frame.shape
    step_size = 20 
    y, x = np.mgrid[step_size/2:h:step_size, step_size/2:w:step_size].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    motion_frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    for (x_i, y_i, fx_i, fy_i) in zip(x, y, fx, fy):
        end_point = (int(x_i + fx_i), int(y_i + fy_i))
        cv2.arrowedLine(motion_frame, (x_i, y_i), end_point, color, thickness, tipLength=0.5)
    
    frame2_resized = resize_frame(frame2, scale_factor)
    motion_frame_resized = resize_frame(motion_frame, scale_factor)
    
    combined_frame = np.vstack((frame2_resized, motion_frame_resized))
    
    cv2.imshow('Original & Motion Vectors', combined_frame)
    out.write(combined_frame)
    
    prvs = next_frame
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()

