import numpy as np
import cv2
from keras.models import model_from_json
import time
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

happy_tracker = []
emotion_dict={0:'anger',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
emotion_to_index={'anger':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5,'surprise':6}
color_dict = {0:(59,59,238),1:(0,205,102),2:(30,30,30),3:(0,215,255),4:(255,255,255),5:(238,134,28),6:(255,102,224)}

with open('model10.json', 'r') as f:
    loaded_model3 = model_from_json(f.read())

loaded_model3.load_weights('mode10.h5')

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

begin = time.time()

while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    start = time.time()
    check = True
    if len(faces) > 5:
        faces = faces[:5]
    for (x,y,w,h) in faces:
        crop_img = gray[int(y+0.05*h):int(y+0.95*h), int(x+0.05*w):int(x+0.95*w)]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        small = cv2.resize(crop_img, dsize = (48,48))
        image3D = np.expand_dims(small,axis = 0)
        image4D = np.expand_dims(image3D, axis = 3)
        image4D3 = np.repeat(image4D, 3, axis=3)
        if time.time()-start >= 0.02 or check == True: 
            # emotions_prob = loaded_model.predict(image4D3)[0]
            # listt = [1 if metric == emotions_prob.max() else 0 for metric in emotions_prob]
            # emotion_index = listt.index(1)
            # emotion = emotion_dict[emotion_index]
            start = time.time()
            check == False
            emotions_prob3 = loaded_model3.predict(image4D3)[0]
            listt3 = [1 if metric == emotions_prob3.max() else 0 for metric in emotions_prob3]
            emotion_index3 = listt3.index(1)
            emotion3 = emotion_dict[emotion_index3]
            
            if emotion3:
                emotion_fin = emotion3
            elif emotion3 == 'happy':
                emotion_fin = 'happy'
            elif emotion3 == 'anger':
                emotion_fin = 'anger'
            elif emotion3 == 'fear':
                emotion_fin = 'fear'
            elif emotion3 == 'neutral':
                emotion_fin = 'neutral'
            elif emotion3 == 'sad':
                emotion_fin = 'sad'
            elif emotion3 == 'disgust':
                emotion_fin = 'disgust'
            elif emotion3 == 'sad':
                emotion_fin = 'sad'
            else:
                emotion_fin = 'neutral'

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_placement  = (34,34)
        fontScale = 1
        fontColor = color_dict[emotion_to_index[emotion_fin]]
        lineType = 4
    
        cv2.putText(frame, 
            '{}'.format(f'{emotion_fin}'), 
            text_placement, 
            font, 
            fontScale,
            lineType)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
