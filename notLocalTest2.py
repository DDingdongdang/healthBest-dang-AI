from flask import Flask                                            # python web framework 
from flask import request                                        # 웹 요청 관련 모듈
from flask import render_template, redirect, url_for, request    # flask에서 필요한 모듈
from flask import jsonify                                        # import JSON을 해도되지만 여기서는 flask 내부에서 지원하는 jsonify를 사용
from keras.preprocessing import image
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from werkzeug.utils import secure_filename	

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

#loading the model i trained and finetuned
my_model = load_model('model_trained.h5')
app = Flask(__name__) # flask를 사용하겠다

@app.route('/') #홈화면에서
def main():
   return render_template('home.html') # home.html을 띄워준다 -> 안드로이드스튜디오랑 연결해줘야할듯?


#function to help in predicting classes of new images loaded from my computer(for now)
#입력값을 받아 모델에 넣어서 예측값을 구하고 이 예측값을 서버에 전달하는 POST 메소드 이용
@app.route("/model", methods=['POST']) 
def model(model, show = True): #이미지 하나 당 하나씩 찾아주기 - 여기서 파라미터들을 어떻게 해야할쥐... 이거 그대로 써도 되는지
    if request.method == 'POST': # POST로 받아오면
        file = request.files['photo'] # 사진 받기 - 형식 안확실 혜빈이가 사진 받아오는 거 구현한거 참고하면 될듯!
        file.save('src/dataset/'+secure_filename(file.filename)) # 로컬에 사진 저장 / secure_filename(): 파일 이름이 안전한지 확인해
        
        # img를 하나하나 크기 맞추기 - 입력값 전처리
        img = utils.load_img(img, grayscale=False, color_mode='rgb', target_size=(299,299))
        img = utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        model = load_model('model_trained.h5') # 분류기 모델 로드
        pred = model.predict(img) # 학습된 분류기에 사진 넣어주기
        index = np.argmax(pred) # 가장 큰 값을 갖는 index 번호 출력
        food_list.sort() 
        pred_value = food_list[index] # 분류한 음식 이름 출력

        return jsonify(pred_value),200 # 예측값을 json 형식으로 내보냄


if __name__ == "__main__": # terminal에서 python 인터프리터로 .py 파일을 실행하면 무조건 이 부분을 찾아 실행합니다.
                           # C의 main
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(port="5000") # app.run을 해줘야 flask 서버가 구동됩니다. 
                            # host="0.0.0.0"은 외부에서 해당 서버 ip 주소 접근이 가능하도록 하는 옵션입니다.
