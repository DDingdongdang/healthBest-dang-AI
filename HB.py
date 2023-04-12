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

food_list = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 
             'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 
             'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 
             'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 
             'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 
             'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 
             'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 
             'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 
             'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
             'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 
             'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 
             'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
             'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 
             'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 
             'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 
             'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
model = load_model('C:/Users/migon/health/healthBest-Dang-AI/model_trained.h5') #loading the model i trained and finetuned
CORS(app)


@app.route("/sendFrame", methods=['POST']) 
def re():
    print(request.method)
    if request.method == 'POST':
        
        foodName = "None" # 음식 이름

        #안드로이드에서 'image'변수에 base64로 변환된 bitmap이미지
        # one_data = request.form['image'];

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
        
        
        #이미지 분석관련 코드 작성
        foodNameData = {"foodName" : foodName}

        photo.save("food.jpg")

        print("===============================================")

        img_path = "food.jpg"
        print("image path reload success")

        # images 의 img를 하나하나 크기 맞추기
        img_data = image.load_img(img_path, grayscale=False, color_mode='rgb', target_size=(299,299))
        img_data = image.img_to_array(img_data)
        img_data = np.expand_dims(img_data, axis=0)
        img_data /= 255.
        print("image preprocess success")


        pred = model.predict(img_data) # 학습된 분류기에 사진 넣어주기
        index = np.argmax(pred) # 가장 큰 값을 갖는 index 번호 출력
        food_list.sort() 
        pred_value = food_list[index]

        # 분류한 음식 이름 출력
        print(pred_value)
        pred_result = {'food name':pred_value}

        # 결과값 json 형식으로 보내줌
        return jsonify(pred_result) 
        


if __name__ == "__main__":
    run_simple('0.0.0.0', 8000, app)