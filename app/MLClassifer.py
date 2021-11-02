from keras_preprocessing import image
from keras.models import model_from_json
from tensorflow.keras.models import load_model
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import cv2
import joblib

class MLClassifer:
    def __init__(self):
        self.class_names_cnn = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        self.class_names_svm = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

        self.class_names_tf=  ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        self.cnn_model=model_from_json(open("models/top_models/fer.json", "r").read())
        self.cnn_model.load_weights('models/top_models/fer.h5')
        self.tf_model=model_from_json(open("models/tf/fer.json", "r").read())
        self.tf_model.load_weights('models/tf/fer.h5')
        self.svm_model=joblib.load("models/svm/emotion_svm.pkl")

    def classifyCnnEmotion(self, faces,imagePath):
        img=cv2.imread(imagePath)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for face in faces:
            x,y,w,h=face["box"]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0
            pridicted_emotion= self.cnn_model.predict(img_pixels)
            cv2.putText(img, self.class_names_cnn[int(np.argmax(pridicted_emotion))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(imagePath,img)
            cv2.waitKey(0)
            face["emotion"]=pridicted_emotion
            
        return faces
    def classifySvmEmotion(self,faces,imagePath):
        img=cv2.imread(imagePath)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for face in faces:

            x,y,w,h=face["box"]
            landmarks_vectorized=face["victorized_landmark"]
            if(landmarks_vectorized!='error'):
                landmarks_vectorized=np.array(landmarks_vectorized).reshape((1, -1)) 
                emo=self.svm_model.predict(landmarks_vectorized)
                
                predicted_emotion = self.class_names_svm[int(emo)]
                cv2.putText(img, self.class_names_svm[int(np.argmax(predicted_emotion))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(imagePath,img)
                cv2.waitKey(0)
                face["emotion"]=emo
        return faces
    def classifyTfEmotion(self,faces,imagePath):
        img=cv2.imread(imagePath)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for face in faces:
            x,y,w,h=face["box"]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0
            pridicted_emotion= self.tf_model.predict(img_pixels)
            cv2.putText(img, self.class_names_tf[int(np.argmax(pridicted_emotion))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(imagePath,img)
            cv2.waitKey(0)
            face["emotion"]=pridicted_emotion
            
        return faces
