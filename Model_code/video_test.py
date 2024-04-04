from ultralytics import YOLO
import cv2
import os
# 모델 입력
model = YOLO("best_yolov8m_film_1st.pt")
# 비디오 경로
video_path = "C:/Dataset/230919_Sorest/TESTSET/50x50/smoke"
# 저장할거면 저장경로
save_path = "D:/bongkj/Projects/230926_sorest_fire/sorest_fire/imgs/2024_03_28_16_19/1"
# 비디오 경로에서 모든 비디오 파일 가져오기
video_list = [] # 여기에 저장
for root, dirs, files in os.walk(video_path):
    for file in files:
        if file.endswith(".mp4"):
            video_list.append(os.path.join(root, file))
# 비디오 루프
for video_idx, video_path in enumerate(video_list):
    # 비디오 캡쳐
    cap = cv2.VideoCapture(video_path)
    # 프레임 카운트 (저장용)
    frame_count = 0
    # 비디오에서 프레임 하나씩 꺼내오기
    while cap.isOpened():
        success, frame = cap.read() # success : 성공여부, frame : 프레임\
        # 실패하면 break(대부분 동영상 끝나는 경우)
        if not success:
            break
        # 프레임 카운트 증가
        frame_count += 1
        # 박스 그릴건지 여부 (저장용)
        plotings = False
        # 예측
        results = model.predict(frame, conf=0.2)
        # 탐지된 박스 정보
        boxes = results[0].boxes.xywh
        confidences = results[0].boxes.conf
        classes = results[0].boxes.cls
        # 컨피던스 넘으면 저장
        for confidence in confidences:
            if confidence > 0.2:
                plotings = True
                continue
        # 컨피던스 넘으면 그리고, 안 넘으면 원본 저장
        if plotings:
            annotated_frame = results[0].plot(conf=True)  # add box and label overlays/
        else:
            annotated_frame = frame.copy()
        # 저장!
        cv2.imwrite(os.path.join(save_path, f'{frame_count:05d}.jpg'), annotated_frame)