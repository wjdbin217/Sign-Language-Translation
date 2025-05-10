import cv2
import sys
import os
import mediapipe as mp
import numpy as np
import json
import time
from tqdm import tqdm

import modules.holistic_module as hm
from modules.utils import createDirectory, Vector_Normalization

createDirectory('dataset/output_video')

save_file_name = "train"
secs_for_action = 10

# 동작 목록
actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]

# 데이터셋 초기화
dataset = {i: [] for i in range(len(actions))}

detector = hm.HolisticDetector(min_detection_confidence=0.3)

video_folder_path = "./dataset/output_video"

# 비디오 폴더 존재 여부 확인
if not os.path.exists(video_folder_path):
    print(f"비디오 폴더 존재하지 않음. : {video_folder_path}")
    sys.exit()

# 모든 비디오 파일 경로 수집
test_target_list = []
for action_folder in os.listdir(video_folder_path):
    action_path = os.path.join(video_folder_path, action_folder)
    if not os.path.isdir(action_path):
        continue
    for video_file in os.listdir(action_path):
        full_video_path = os.path.join(action_path, video_file)
        if os.path.isfile(full_video_path):
            test_target_list.append(full_video_path)

print("---------- 비디오 목록 시작  ----------")
test_target_list = sorted(test_target_list, key=lambda x: os.path.basename(os.path.dirname(x)), reverse=True)
print(test_target_list)
print("----------  비디오 목록 끝  ----------\n")

created_time = int(time.time())

# 전체 비디오 리스트에 tqdm 프로그레스 바 적용
for target in tqdm(test_target_list, desc="전체 비디오 처리", unit="video"):
    data = []
    
    # 경로에서 동작 라벨 추출
    action_label = os.path.basename(os.path.dirname(target))
    try:
        idx = actions.index(action_label)
    except ValueError:
        print(f"동작 '{action_label}'을 동작 목록에서 찾을 수 없음.  비디오를 건너뜀. : {target}")
        continue            # 라벨이 없으면 비디오 건너뜀
    
    print("현재 스트리밍 중 :", target)
    cap = cv2.VideoCapture(target)
    
    # 비디오 열기 확인
    if not cap.isOpened():
        print(f"비디오를 열 수 없음 : {target}.건너 뜀.:")
        continue               # 다음 비디오로 이동

    # 비디오 속성 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"비디오 속성 - 너비 : {w}, 높이 : {h}, FPS : {fps}")
    
    if fps > 0:
        delay = int(1000 / fps)
    else:
        delay = 1            # 최소 딜레이임.
    
    # 비디오의 총 프레임 수
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # tqdm 프로그레스 바를 프레임 처리에 적용하기.
    with tqdm(total=total_frames, desc=f"프레임 처리 ({action_label})", unit="frame", leave=False) as pbar:
        while True:
            ret, img = cap.read()
            
            if not ret:
                break 
            
            img = detector.findHolistic(img, draw=False)
            
            # 오른손 랜드마크 추출하기.
            _, right_hand_lmList = detector.findRighthandLandmark(img)
            
            if right_hand_lmList:

                # 오른손 21개 랜드마크 (x, y)
                joint = np.zeros((21, 2))
                
                for j, lm in enumerate(right_hand_lmList.landmark):
                    joint[j] = [lm.x, lm.y]
                
                vector, angle_label = Vector_Normalization(joint)
                
                angle_label = np.append(angle_label, idx)
                
                # 정규화된 벡터와 라벨 결합.
                d = np.concatenate([vector.flatten(), angle_label.flatten()])
                
                data.append(d)
            
            # 프로그레스 바 업데이트
            pbar.update(1)

            # esc 키로 종료 (선택 사항, 비활성화 가능)
            # 하지만 비디오를 표시하지 않으므로 필요 없을 듯....

            # if cv2.waitKey(1) & 0xFF == 27:
            #     print("ESC 키가 눌려서 종료함.")
            #     cap.release()
            #     sys.exit()
    
    print("\n---------- 비디오 스트리밍 종료 ----------")
    cap.release()
    # 창을 생성하지 않았으므로 destroyAllWindows는 필요 없음.
    
    data = np.array(data)
    
    # 시퀀스 데이터 생성
    if len(data) >= secs_for_action:
        for seq in range(len(data) - secs_for_action + 1):
            dataset[idx].append(data[seq:seq + secs_for_action])
    else:
        print(f"비디오에서 시퀀스 길이 {secs_for_action}를 충족할 만큼 데이터가 부족함. : {target}")

# 데이터셋 저장
print("\n---------- 데이터셋 저장 시작 ----------")
for i in range(len(actions)):
    if dataset[i]:
        save_path = os.path.join('dataset', f'seq_{actions[i]}_{created_time}.npy')
        np.save(save_path, np.array(dataset[i]))
        print(f"동작 '{actions[i]}'에 대해 {len(dataset[i])}개의 시퀀스를 '{save_path}'에 저장헸음.")
    else:
        print(f"동작 '{actions[i]}'에 대한 데이터가 없으니까 저장을 건너뜀.")

print("\n---------- 데이터셋 저장 종료 ----------")