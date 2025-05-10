import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import math
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

# 폰트 설정
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

# 인식할 동작 리스트
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
            'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
secs_for_action = 10       # 시퀀스 길이임.

detector = hm.HolisticDetector(min_detection_confidence=0.3)
# Keras 모델 로드
model = load_model("./best_model.keras")

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # holistic 모델을 사용하여 이미지 처리
    img = detector.findHolistic(img, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:

        joint = np.zeros((21, 2))        # 오른손 랜드마크는 21개임.
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        # 벡터 정규화
        vector, angle_label = Vector_Normalization(joint)

        d = np.concatenate([vector.flatten(), angle_label.flatten()])
        seq.append(d)

        if len(seq) < secs_for_action:    # 시퀀스 길이가 10에 도달하지 않으면 다음 프레임으로 넘어감.
            continue

        input_data = np.expand_dims(np.array(seq[-secs_for_action:], dtype=np.float32), axis=0)

        # Keras 모델을 활용한 예측
        y_pred = model.predict(input_data)
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        if conf < 0.8:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action

        # 텍스트를 이미지의 정가운데에 출력
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        text = f'{action.upper()}'

        # 텍스트 크기 측정
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]


        img_width, img_height = img_pil.size       # 이미지 크기 가져오기
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2

        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
