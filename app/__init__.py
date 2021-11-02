from flask import Flask


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

TEMP_FOLDER = './tmp'
DATA_FOLDER ="data_dir"
ROOT_URL="http://127.0.0.1:5000/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config["DATA_DIR"]= DATA_FOLDER
app.config["ROOT_URL"] = ROOT_URL

from app import views
