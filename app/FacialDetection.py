from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from keras.preprocessing import image
from matplotlib.patches import Rectangle
import cv2
import numpy as np

class FacialDetection:
    def detectFaces(self, imagePath):
        img = cv2.imread(imagePath)
        face_detector = MTCNN()
        faces = face_detector.detect_faces(img)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if len(faces)==0:
            return 0
        i=0
        for face in faces:
            path=imagePath[0:-4]+str(i)
            x,y,w,h=face["box"]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (192, 192))
            img_pixels = image.img_to_array(roi_gray)
            pathName=path+str(i)+".jpg"
            cv2.imwrite(pathName,img_pixels)
            cv2.waitKey(0)
            i=i+1
            face["tmp_path"]=pathName
        return faces

    def draw_image_with_boxes(self, filename, result_list):
        data = cv2.imread(filename)
        pyplot.imshow(data)
        ax = pyplot.gca()
        for result in result_list:
            x, y, width, height = result['box']
            rect = Rectangle((x, y), width, height, fill=False, color='red')
            ax.add_patch(rect)
            ax.text((x), (y), 'middle',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='red',
                    transform=ax.transAxes)
        ax.set_axis_off()
        
        return data
