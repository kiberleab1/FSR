import os
import numpy as np
import shutil
class OutputBuilder:
    def __init__(self):
         self.cnn_labels= ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
         self.svm_labels= ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
         self.tf_labels=  ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    def buildJson(self,pridictedEmotions,algorithm_type,img_path):
        outputJson=[]
        root_url="http://127.0.0.1:5000/"
        if algorithm_type=="CNN":
            for face in pridictedEmotions:
                pridictedEmotion=face["emotion"]
                
                json={}
                p=np.max(pridictedEmotion)
                emotion=self.cnn_labels[int(np.argmax(pridictedEmotion))]
                json["emotion"]=emotion
                emotionConfidence={}
                i=0
            
                    
                for emotionProb in pridictedEmotion.tolist()[0]:
                    emotionConfidence[self.cnn_labels[i]]=emotionProb
                    i=i+1

                json["detail"]=emotionConfidence
                shutil.move(face["tmp_path"],"./data_dir/"+emotion+""+face["tmp_path"][5:])
                json["link"]=root_url+"api/data/image?emotion="+emotion+"&img_id="+face["tmp_path"][6:]
                outputJson.append(json)
        elif algorithm_type=="SVM":
            for face in pridictedEmotions:
                landmarks_vectorized=face["victorized_landmark"]
                if(landmarks_vectorized!='error'):
                    pridictedEmotion=face["emotion"]
                    
                    json={}
                    emotion=self.svm_labels[int(pridictedEmotion)]
                    json["emotion"]=emotion
                    
                    shutil.move(face["tmp_path"],"./data_dir/"+emotion+""+face["tmp_path"][5:])
                    json["link"]=root_url+"api/data/image?emotion="+emotion+"&img_id="+face["tmp_path"][6:]
                    outputJson.append(json)
        elif algorithm_type=="TF":
            
            for face in pridictedEmotions:
                pridictedEmotion=face["emotion"]
                
                json={}
                p=np.max(pridictedEmotion)
                emotion=self.cnn_labels[int(np.argmax(pridictedEmotion))]
                json["emotion"]=emotion
                emotionConfidence={}
                i=0
            
                    
                for emotionProb in pridictedEmotion.tolist()[0]:
                    emotionConfidence[self.cnn_labels[i]]=emotionProb
                    i=i+1

                json["detail"]=emotionConfidence
                shutil.move(face["tmp_path"],"./data_dir/"+emotion+""+face["tmp_path"][5:])
                json["link"]=root_url+"api/data/image?emotion="+emotion+"&img_id="+face["tmp_path"][6:]
                outputJson.append(json)
        json={}

        shutil.move(img_path,"./data_dir/uploads"+img_path[5:])
        json["labeldImage"]=root_url+"api/data/image?emotion=uploads&img_id="+img_path[6:]
        outputJson.append(json)
            
         
        return outputJson