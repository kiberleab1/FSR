
from app import ALLOWED_EXTENSIONS, ProcessHandler, app
from flask import json, request, jsonify ,send_from_directory
from werkzeug.utils import secure_filename
import os
import icecream as ic 
from pathlib import Path
@app.route("/")
def index():
    return jsonify({
        "POST":"/api/fsr"
    })

@app.route("/api/home", methods=["POST"])
def home():
    photo=request.files["photo"]
    processHandler=ProcessHandler.ProcessHandler()
    photoname=secure_filename(photo.filename)
    photo.save(os.path.join(app.config["TEMP_FOLDER"],photoname))
    path=os.path.join(app.config["TEMP_FOLDER"],photoname)
    isValid=processHandler.validateInputHome(request)
    if isValid!=True:
        return jsonify(isValid), 301
    algorithm_type=request.form["algorithm_type"]
    detected_faces=processHandler.detectFaces(path)
    if detected_faces==0:
        return jsonify("No Face Detected"), 400
    emotions=processHandler.detectEmotions(detected_faces,path,algroithem_type=algorithm_type)
    output=processHandler.buidOutputbuilderWithImageLabel(emotions,algorithm_type,path)
    return jsonify(output)

@app.route("/api/data", methods=["GET"])
def getData():
    dir_list=os.listdir(app.config["DATA_DIR"])
    root_url=app.config["ROOT_URL"]
    links={}
    for dir_name in dir_list:
        links[dir_name]=root_url+"api/data/single?emotion="+dir_name
        
    return jsonify(links)

@app.route("/api/data/single", methods=["GET"])
def getSingleEmotionData():
    dir_list=os.listdir(app.config["DATA_DIR"])
    
    if (request.args.get("emotion") == None or request.args.get("emotion") not in dir_list ):
        return jsonify("Emotion not selected or unavaliable emotion selected"), 400
    selected_emotion=request.args.get("emotion")

    images_path=app.config["DATA_DIR"]+"/"+selected_emotion
    url=app.config["ROOT_URL"]+"api/data/image?emotion="+selected_emotion+"&img_id="
    image_list=os.listdir(images_path)
    image_list=[url+x for x in image_list]
    return jsonify(image_list)

@app.route("/api/data/image", methods=["GET"])
def getImage():
    dir_list=os.listdir(app.config["DATA_DIR"])
    if (request.args.get("emotion") == None or request.args.get("emotion") not in dir_list ):
        return jsonify("Emotion not selected or unavaliable emotion selected"), 400

    selected_emotion=request.args.get("emotion")
    images_path=app.config["DATA_DIR"]+"/"+selected_emotion
    image_list=os.listdir(images_path)
    if (request.args.get("img_id") == None or request.args.get("img_id") not in image_list ):
        return jsonify("Image not selected or Image emotion selected"), 400
    img_id=request.args.get("img_id") 
    path=images_path+"/"
    try:
        path=os.path.abspath(os.path.join(app.root_path,os.pardir,images_path))
        return send_from_directory(path,img_id)
    except Exception as e:
        return str(e)

