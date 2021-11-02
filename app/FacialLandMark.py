
import os, shutil, sys, time, re, glob
import dlib
import itertools
import operator
import cv2, glob, random, sys, math, numpy as np, dlib, itertools
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
import joblib
from io import StringIO

class FacialLandMark:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/svm/shape_predictor_68_face_landmarks.dat")
    


    
    def get_landmarks(self,image):
        detections = self.detector(image, 1)
        for k,d in enumerate(detections): 
            shape = self.predictor(image, d) 
            xlist = []
            ylist = []
            for i in range(1,68): 
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
                
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist] 
            ycentral = [(y-ymean) for y in ylist]

            if xlist[26] == xlist[29]: 
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            landmarks_vectorised = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append(anglerelative)

        if len(detections) < 1: 
            landmarks_vectorised = "error"
        return landmarks_vectorised

