![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/662ef95c-3051-4d38-b2b9-67d5a3d553de)

# 🖥 AI Team
수행자: 김민수, 김소희, 이수연, 오재민
---
## 💡 Introduction
시각 장애 보조를 위한 딥러닝 기반 장애물 인식
Deep Learning Based Obstacle Detection

## 🕰️ 수행 기간
2024.03.04 ~ 2024.04.03

### **🕹 수행 방법▪도구**

- YOLO-v8를 사용한 Object Detecting (객체 탐지)
- IDE : Jupyter Notebook, Google Colab
- Tool : Python
- library : pytorch, opencv, sklearn, matplotlib, numpy, pandas

## 📌 1.프로젝트 개요
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/09a0e9d1-1b94-442c-a86f-12f60336ef5a)

시각장애인은 현재에도 다양한 보행의 어려움을 겪고 있음
이에 보행에 위협 요소인 각종 장애물(자동차, 사람, 가로등, 가로수 등)과 파손 등으로 위험한 보행 노면에 대한 데이터 셋을 구축하여 이동권 신장을 위한 인공지능 서비스 개발의 기반을 구축하고자 함
개발된 인도 보행 영상 데이터 셋은 기존 자율주행차량 데이터 셋과 달리 국내 보행자 현실에 맞는 인도를 중심으로 구축될 부분이 기존 타 데이터들과 차별화 됨

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/3556f0dc-cfbb-4337-ad70-0d7e0665939a)

## 🎲 2.활용 방안
구축된 데이터 셋은 시각장애인 보행 위험 알림 서비스, 휠체어 사용자를 위한 안전 경로 안내 서비스 등
장애인 보행권 확보를 위한 AI 서비스 개발 뿐만 아니라 공공 시설 보행 안전성 강화를 위한
노면 안정성 모니터링 서비스 등 다양한 서비스 개발에 활용이 가능
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/fc114c86-1aa1-4d0f-ba0f-72f7db950bd3)


---

## 📸 3.데이터 수집
### 데이터셋
[http://drive.google.com/drive/folders/1Lk6_Ndl8FSHGnvlY9ofgzU9WtTlkg-VR?usp=sharing](https://drive.google.com/drive/folders/1-HItjUQ7jjKbqD5q4mCiOWI8DqfziC9v?usp=sharing)
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/056c1805-7b1d-44bd-8e1e-79b5362cfd80)

### 데이터 정보
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/d0cdf8df-d93b-4f1f-979f-7db3d62427d9)


## 🕹️ 4.YOLO v8 학습 과정 요약 | Overview
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/0e3e0c50-d05d-421a-b5d0-7250879b5c05)

- 1차학습 : Train 데이터셋 과 Val 데이터셋 동일하게 진행, 100epoch
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/ff0aad25-7f6a-43d2-a152-a52c0fcf69df)
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/c66172b4-1b2d-41d3-8c65-d5a7c499a1e4)

- 2차학습 : Train 데이터셋 과 Val 데이터셋 중복 되지 않도록 분리 , class 목록 40여개
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/00020328-46d9-42a7-b3d5-bc52d4280eae)
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/31433b01-485a-4898-aa2e-245bdb4ab248)

- 3차학습 : 필요없는 클래스 제거 후 재 학습 10epoch
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/88d6fb0b-eacf-47b2-8f18-1cdedf8005f2)
컨퓨전 행렬 (Confusion matrix) 확인 시
일부 컬럼을 제외하고 YOLO performs 수치를 확인할 수 있음
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/f883f38e-fe74-4e79-95cc-f13047fb5236)

F1-Confidence Curve
Person > Motorcycle > Bicycle > Obstacle 순으로 Confidence 값이 높게 출력 됨 
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/6bd838cb-f06a-4e70-b744-71ac5bc53f77)

----------------
(추가 전처리 진행)
🚫 3차 학습 영상 결과 확인 시 성능이 개선되지 않아 라벨링 된 데이터를 다시 체크해 보니
오토라벨링으로 인한 라벨명 오류 & 박스의 위치 부정확이 확인되어, 전면 수정 (추가 전처리) 진행

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/5ec92893-7eff-4836-88be-65aff3213d78)

- 4차 학습 : 라벨링 수정 후 마지막 모델 학습 - 4차 학습 결과 ( 100 epoch )
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/f2f893f5-44f5-4cd9-8de7-6eec6f7dcdb2)

4차 학습의 컨퓨전 행렬 (Confusion matrix) 확인 시 3차 학습 대비 성능이 소폭 개선됨 
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/085372ef-333b-4d52-873b-cb1d419b4ee4)

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/db97a054-a398-456b-8805-ef3ea70f5699)

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/a5159261-17e2-46eb-ad40-6585aa4b38d2)

## 🥇 5.학습 결과 
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/cb22df9d-12f6-4369-a590-e817dd96a1c2)

## 알고리즘
3D 렌더링에서 관찰점을 중앙으로 모으는 것을 "중심 정렬" 또는 "카메라 중심 정렬"이라고 한다
관찰자 시점에서 중심 정렬된 네모 박스는 사용자의 시선을 대변하여 위험한 물체나 상황을
시각적으로 명확하게 보여줌으로써 대응 능력을 향상시켜 줄 것으로 가정함

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/c49c9eff-9431-4ecf-b797-9aa384a388b2)

중심 정렬 네모박스를 기준으로 약 30% 이상 가까워지는 경우,
[객체 라벨명]이 포함된 [Danger] 경고 메시지를 표시하여 위험을 감지할 수 있도록 개발

![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/bf092a68-a0cc-4ec6-b857-483ceb132614)

## 🤹‍♂️ 6.테스트 영상

* 낮 영상 TEST 
  [Video](https://drive.google.com/file/d/1RnfllB1eP3Zpma1ZafQjPef4hU3V70aQ/view?usp=sharing)
* 밤 영상 TEST 
  [Video](https://drive.google.com/file/d/1a0xsWnle6wHmJQEReLGeP0D-5rlqjI5E/view?resourcekey)

## 🔫 7.회고 & 확장 방안
![image](https://github.com/somoon0422/Deep-Learning-Based-Obstacle-Detection/assets/116736551/6ba37e1f-0d03-4dc3-9ab5-62a227fedb43)
