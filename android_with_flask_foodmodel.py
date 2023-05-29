from flask import Flask                                            # python web framework 
from flask import render_template, redirect, url_for, request    # flask에서 필요한 모듈
from flask import jsonify                                        # import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용
from keras.preprocessing import image
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64
from keras.models import load_model
from werkzeug.utils import secure_filename	
from flask_cors import CORS
from PIL import Image
from werkzeug.serving import run_simple

import foodmodel_module as fmm

food_dict = fmm.load_foodDict()
model = fmm.load_foodmodel()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
CORS(app)


@app.route("/sendFrame", methods=['POST']) 
def re():
    print(request.method)
    if request.method == 'POST':
        
        #foodName = "None" # 음식 이름

        #안드로이드에서 'image'변수에 base64로 변환된 bitmap이미지
        one_data = request.form['image']

        #웹에서 base64로 인코딩된 이미지 정보 가져오기
        _, one_data = request.form['image'].split(',') 
        print("Success to get incoding image from user") # debugging
        print('incoding image:', one_data[:10]) # base64 코드 앞쪽 10자리만 확인

        #base64로 인코딩된 이미지 데이터를 디코딩하여 byte형태로 변환
        imgdata = base64.b64decode(one_data)
        print("Success to decode base64 code") # debugging

        #byte형태의 이미지 데이터를 이미지로 변환
        print("Success to get image data") # debugging
        photo = Image.open(io.BytesIO(imgdata))
        if photo is not None :
            foodName = "food"

        print(type(photo)) 
        photoArray = np.array(photo) # numpy 형식 배열로 만들어줌
        print(photoArray.shape) # shape를 사용하기 위해서
        
        #이미지 분석관련 코드 작성
        #foodNameData = {"foodName" : foodName}

        photo.save("food.jpg")

        print("===============================================")

        img_path = "food.jpg"
        print("image path reload success")

        # images 의 img를 하나하나 크기 맞추기
        img_data = image.load_img(img_path, grayscale=False, color_mode='rgb', target_size=(fmm.img_height,fmm.img_width))
        img_data = image.img_to_array(img_data)
        print("image preprocess success")

        pred_value = fmm.predict_one_food(food_dict, model, img_data)

        # 분류한 음식 이름 출력
        print(pred_value)
        pred_result = {'food name':pred_value}

        # 결과값 json 형식으로 보내줌
        return jsonify(pred_result) 
        


if __name__ == "__main__":
    run_simple('0.0.0.0', 8000, app)