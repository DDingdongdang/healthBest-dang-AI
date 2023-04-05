from flask import Flask, redirect, request, Response
import base64
import io
from PIL import Image
from flask_cors import CORS
from werkzeug.serving import run_simple

app = Flask(__name__)
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
        image = Image.open(io.BytesIO(imgdata))
        if image is not None :
            foodName = "food"
        
        #이미지 사이즈 조정
        img_resize = image.resize((int(image.width / 2), int(image.height / 2)))
        
        img_resize.show() # 받은 이미지 확인
        # display(img_resize)
        
        #이미지 분석관련 코드 작성
        foodNameData = {"foodName" : foodName}
        
    #결과값 리턴    
    return foodNameData

if __name__ == "__main__":
    run_simple('0.0.0.0', 8000, app)