import cv2
import sys, os
import time
import mediapipe as mp
from modules.utils import createDirectory
import numpy as np
from PIL import ImageFont, ImageDraw, Image

fontpath = "./fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

createDirectory('dataset')

actions = ['ㄱ']
# actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# actions = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
# actions = ['ㅗ', 'ㅛ', 'ㅜ']
# actions = ['ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

secs_for_action = 10

# mediapipe 모델델
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())

# 열렸는지 확인
if not cap.isOpened():
    print("카메라 안 열림.")
    sys.exit()


# 웹캠의 속성 값을 받아오기
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps != 0:
    delay = round(1000 / fps)
else:
    delay = round(1000 / 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 프레임을 받아와서 저장.
while cap.isOpened():
    for idx, action in enumerate(actions):
        
        os.makedirs(f'./dataset/output_video/{action}', exist_ok=True)

        videoFolderPath = f'./dataset/output_video/{action}'
        videoList = sorted(os.listdir(videoFolderPath), key=lambda x: int(x[x.find("_")+1:x.find(".")]))
      
        if len(videoList) == 0:
            take = 1
        else:
            f = videoList[-1].find("_")
            e = videoList[-1].find(".")
            take = int(videoList[-1][f+1:e]) + 1

        saved_video_path = f'./dataset/output_video/{action}/{action}_{take}.avi'

        out = cv2.VideoWriter(saved_video_path, fourcc, fps, (w, h))

        # 대기 시간 동안 텍스트 표시
        wait_time = 4          # 대기 시간(초 단위임.)
        for remaining_time in range(wait_time, 0, -1):
            ret, img = cap.read()
            if not ret:
                break

            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            # 텍스트 내용 설정
            text = f'{action.upper()} 입력 대기중\n{remaining_time}초 후 시작'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (w - text_width) // 2
            text_y = (h - text_height) // 2

            # 검은색 박스 그리기
            box_margin = 10
            box_coords = (
                text_x - box_margin,
                text_y - box_margin,
                text_x + text_width + box_margin,
                text_y + text_height + box_margin
            )
            draw.rectangle(box_coords, fill=(0, 0, 0, 200))     # 반투명 검은색 박스

            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))



            # 배열 변환 및 화면 표시
            img = np.array(img_pil)
            cv2.imshow('img', img)

            # 1초 대기하기
            if cv2.waitKey(1000) == 27:        # ESC 키 누르면 종료함.
                break

        start_time = time.time()

        # 10초 동안 동작 캡처
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                break
            
            
            out.write(img)    # 비디오 녹화

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)

            # esc를 누르면 강제 종료됨.
            if cv2.waitKey(delay) == 27: 
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
