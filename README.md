# 🧥 FashionMNIST CNN 이미지 분류기

PyTorch와 Gradio를 사용하여 구현한 패션 아이템 이미지 분류 데모입니다.

## 📁 프로젝트 구조

```
CNN/
├── README.md
├── requirements.txt
├── model.py               # CNN 모델 정의
├── train.py               # 모델 학습 스크립트
├── dataset.py             # CSV 데이터 로더
├── utils.py               # 유틸리티 함수
├── web_app/
│   └── app.py             # Gradio 웹 애플리케이션
└── data/
    ├── fashion-mnist_train.csv
    └── fashion-mnist_test.csv
```

## 🚀 시작하기

### 1. 가상환경 생성 및 활성화 (권장)

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
# 가상환경 활성화 (Windows)
# venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 모델 학습

> 이 과정은 최초 한 번만 실행하면 됩니다.

`train.py`를 실행하여 CSV 데이터로 모델을 학습시키고, `fashion_mnist_cnn.pth` 파일을 생성합니다.

```bash
python train.py
```

### 4. 웹 애플리케이션 실행

학습이 완료되면, 아래 명령어로 Gradio 웹 앱을 실행합니다.

```bash
cd web_app
python app.py
```

터미널에 표시되는 URL (예: `http://127.0.0.1:7860`)을 웹 브라우저에서 열어주세요.
