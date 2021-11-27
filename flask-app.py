#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 03:03:17 2021

@author: akhil
"""
from flask import render_template, request, Response, make_response
from flask import Flask

import numpy as np
import matplotlib.pyplot as plt

from mrcnn_model import MRCNN
import tensorflow as tf

import os
from PIL import Image
import base64
import json
import gzip

app = Flask(__name__)  

ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TEMP_FILE = "tmp.jpg"
DETECTION_IMAGE_DPI=100

def load_model():
    # Load the pre-trained Keras model
    global model
    model = MRCNN()
    
    # This is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()
    
    
def image_resize(img):
    basewidth = 256
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((basewidth,hsize), Image.ANTIALIAS)


def get_detection_result(img_filepath):
    print(img_filepath)
    
    base=os.path.basename(img_filepath)
    filename = os.path.splitext(base)[0]
    
    print("filename in get_detection_result ",filename)
    
    img = Image.open(img_filepath)
    resized_img = image_resize(img)
    img_arr=np.array(img)
    
    fig = plt.figure(figsize = (10,5))
    
    ax1 = fig.add_subplot(121)
    
    # plt.xticks([])
    # plt.yticks([])
   
    ax1.xaxis.set_visible(False) # same for y axis.
    ax1.yaxis.set_visible(False)
    # https://www.delftstack.com/howto/matplotlib/how-to-turn-off-the-axes-for-subplots-in-matplotlib/
    
    ax1.title.set_text('Original Image')
    ax1.imshow(resized_img, cmap = 'gray');
    
    ax2 = fig.add_subplot(122)
    
    img_arr = np.array(img).astype(float) 
            
    preds=None
    print("Detection for",filename)
    with graph.as_default():            
        preds = model.detect(img_arr, fig)
        ax2.title.set_text('Detection Image')
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
    
    plt.savefig(TEMP_FILE,dpi=DETECTION_IMAGE_DPI)
    
    with open(TEMP_FILE, "rb") as f:
        im_bytes = f.read()        
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    
    data = {"success": True}
    data["rois"]=preds["rois"].tolist()
    data["class_ids"] = preds["class_ids"].tolist()
    data["scores"] = preds["scores"].tolist()
    data["image"] = im_b64
    return data


@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")


@app.route('/detectRestGZip', methods = ['POST'])  
def detect_rest_gzip():
    if 'file' not in request.files:
        response = {
            'message': "No file uploaded within the POST body."
        }
        return json.dumps(response), 400
    
    uploaded_file = request.files['file']
    
    filename = uploaded_file.filename
    
    img_filepath = os.path.join(IMAGE_DIR, filename)
    uploaded_file.save(img_filepath)
    
    try:
        data = get_detection_result(img_filepath)
        content = gzip.compress(json.dumps(data, indent=4).encode('utf8'), 5)
        response = make_response(content)
        response.headers['Content-length'] = len(content)
        response.headers['Content-Encoding'] = 'gzip'
        return response
    except:
        print("Something went wrong")
        return json("Error"), 400
    finally:
        if os.path.exists(os.path.join(IMAGE_DIR, filename)):
            os.remove(os.path.join(IMAGE_DIR, filename))        
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

@app.route('/detectRest', methods = ['POST'])  
def detect_rest():
    if 'file' not in request.files:
        response = {
            'message': "No file uploaded within the POST body."
        }
        return json.dumps(response), 400
    
    uploaded_file = request.files['file']
    
    filename = uploaded_file.filename
    
    img_filepath = os.path.join(IMAGE_DIR, filename)
    uploaded_file.save(img_filepath)
    
    try:
        data = get_detection_result(img_filepath)
        return Response(json.dumps(data, indent=4), mimetype='application/json charset=utf-8')
    except:
        print("Something went wrong")
        return json("Error"), 400
    finally:
        if os.path.exists(os.path.join(IMAGE_DIR, filename)):
            os.remove(os.path.join(IMAGE_DIR, filename))        
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
 
@app.route('/detectUI', methods = ['POST'])  
def detect_page():  
    if 'file' not in request.files:
        response = {
            'message': "No file uploaded within the POST body."
        }
        return json.dumps(response), 400
    
    uploaded_file = request.files['file']
    
    filename = uploaded_file.filename
    
    img_filepath = os.path.join(IMAGE_DIR, filename)
    uploaded_file.save(img_filepath)
    
    try:
        data = get_detection_result(img_filepath)
        return render_template("detection.html", var = data)
    except:
        print("Something went wrong")
        return json("Error"), 400
    finally:
        os.remove(os.path.join(IMAGE_DIR, filename))
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)
    
  
if __name__ == '__main__':  
    load_model()
    app.run(debug = True , use_reloader=False)
    #