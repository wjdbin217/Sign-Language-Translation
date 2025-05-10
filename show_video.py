import cv2
import sys, os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

fontpath = "./fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

videoFolderPath = "./dataset/output_video"
videoTestList = os.listdir(videoFolderPath)

testTargetList =[]

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    if videoPath == ".DS_Store":
        continue
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)



print("---------- 비디오 리스트 start -------------")
testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)
print(testTargetList)
print("----------  비디오 리스트 end  ----------\n")

for target in testTargetList:
    print("streaming : ", target)
    cap = cv2.VideoCapture(target)

    # 열렸는지 확인해야됨.
    if not cap.isOpened():
        print("카메라가 열기 실패패.")
        sys.exit()

    # 웹캠의 속성 값을 받아오기
    
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # 정수 형태로 변환하기 위해 round
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)    # 카메라에 따라 값이 정상적, 비정상적

    if fps != 0:
        delay = round(1000/fps)
    else:
        delay = round(1000/30)

    # 프레임을 받아와서 저장
    while True:
        ret, img = cap.read()

        if not ret:
            break

        cv2.rectangle(img, (0,0), (w, 70), (245, 117, 16), -1)



        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((15,20), target[23:], font=font, fill=(0, 0, 0))
        img = np.array(img_pil)


        cv2.imshow('img', img)
        cv2.waitKey(delay)

        # esc를 누르면 강제 종료됨.
        if cv2.waitKey(delay) == 27: 
            break


    cap.release()
    cv2.destroyAllWindows()