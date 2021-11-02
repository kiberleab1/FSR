import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from icecream import ic
from enum import Enum as enum


class ImagePreProcesser:
    def resize(self, image, width=48, height=48, type="path"):
        if(type == "path"):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        elif(type == 'cv2array'):
            img = image
        else:
            return
        dimension = (width, height)
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_LINEAR)
        return img

    def removeNoise(self, image, type="path"):
        if(type == "path"):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        elif(type == 'cv2array'):
            img = image
        else:
            return
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def cropImages(self, image, points=(218, 218+78,500, 500+67), type="path"):
        if(type == "path"):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        elif(type == 'cv2array'):
            img = image
        else:
            return
        cropped_image = img[points[0]:points[1], points[2]:points[3]]

        return cropped_image

    def display(self, image, type='path'):
        if(type == "path"):
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        elif(type == 'cv2array'):
            img = image
        else:
            return
        plt.imshow(img)
        plt.title("Facial Sentimintal Recognition")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return

    def loadImagesFromPath(self, path):
        image_files = np.array([os.path.join(path, file)
                                for file in os.listdir(path) if file.endswith('.jpg') or file.endswith('.png')])
        image_files = np.sort(image_files)
        return image_files

    def saveImageToDir(self, image, path, img_name):
        cv2.imwrite(path + img_name+".jpg",image)
        cv2.waitKey(0)
        return 0
    def preProcessForSvm(self,image):
        image= cv2.imread(image, cv2.IMREAD_UNCHANGED)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(image)
        return clahe_image
