from numpy.lib.function_base import vectorize
from app.OutputBuilder import OutputBuilder
from icecream import ic 
import numpy as np
from app import ImagePreProcesser, app
from app import FacialDetection , MLClassifer, OutputBuilder, FacialLandMark
import cv2
import os
class ProcessHandler:
    def __init__(self):
        self.ipp= ImagePreProcesser.ImagePreProcesser()
        self.facialDetection=FacialDetection.FacialDetection()
        self.mlClassifier=MLClassifer.MLClassifer()
        self.outputBuilder=OutputBuilder.OutputBuilder()
        self.facialLandMark=FacialLandMark.FacialLandMark()
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    def ResizeImageAndRemoveNoiseImage(self,image):
        img=self.ipp.resize(image)
        img=self.ipp.removeNoise(img,type='cv2array')
        return img

    def detectFaces(self,image):
       
        detectFaces=self.facialDetection.detectFaces(image)
        return detectFaces
    def detectEmotions(self,faces,image,algroithem_type):
        if(algroithem_type=="CNN"):
            return self.mlClassifier.classifyCnnEmotion(faces,image)
        elif(algroithem_type=="SVM"):
            for face in faces:
                
                img=self.ipp.preProcessForSvm(image=face["tmp_path"])
                victorized_landmarks=self.facialLandMark.get_landmarks(img)
                face["victorized_landmark"]=victorized_landmarks
            return self.mlClassifier.classifySvmEmotion(faces,image)
        elif(algroithem_type=="TF"):
            return self.mlClassifier.classifyTfEmotion(faces,image)
            

    def buidOutputbuilderWithImageLabel(self,faces,algorithm_type,imgPath):
        return self.outputBuilder.buildJson(faces,algorithm_type,imgPath)



    def validateInputHome(self,req):
        algorithm_types_list=["CNN","SVM","TF"]
        if req.method== 'POST':
            if 'photo' not in req.files or "algorithm_type" not in req.form:
                return "Missing photo or/and algorithm_type fields"
            
            photo=req.files["photo"]
            algorithm_type=req.form["algorithm_type"]

            if photo.filename== '':
                return "Photo not selected", 400
            if algorithm_type=='' or algorithm_type not in algorithm_types_list:
                return "Empty algoritm type or invalid alogrithm please select either CNN SVM or TF", 400

            if photo and self.allowed_file(photo.filename):
                return "Invalid File Type", 400
        return True
    def allowed_file(self,filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() not in self.ALLOWED_EXTENSIONS
