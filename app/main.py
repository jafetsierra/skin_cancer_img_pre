from fastapi import FastAPI, File, UploadFile
import json
import numpy as np
from PIL import Image
import requests

app = FastAPI()

def make_prediction(file):
    SERVER_URL = 'https://skin-cancer--pred-api.herokuapp.com/v1/models/skin_cancer_model:predict'
    img        = np.array(Image.open(file).resize((224,224)),dtype=np.float32)
    tensor     = img = np.expand_dims(img,0)
    input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": tensor.tolist(),
    })
 
    response = requests.post(SERVER_URL, data=input_data_json)
    response.raise_for_status() # raise an exception in case of error
    response = response.json()
    y_proba = np.array(response["predictions"])
    return {
        "prediction": y_proba[0].tolist()
    }


@app.post("/")
def file_process(file: UploadFile = File(...)):
    return {
        "response" : make_prediction(file.file)
    }