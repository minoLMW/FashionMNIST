import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# 프로젝트 루트를 시스템 경로에 추가하여 model 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CNN, LABEL_TAGS

class FashionClassifier:
    def __init__(self, model_path='../fashion_mnist_cnn.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        """학습된 모델 가중치를 로드하고 모델을 초기화합니다."""
        model = CNN().to(self.device)
        try:
            # 모델 가중치 로드
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"✅ 모델 로딩 성공: {model_path}")
            return model
        except FileNotFoundError:
            print(f"❌ 모델 파일({model_path})을 찾을 수 없습니다.")
            print("먼저 `python train.py`를 실행하여 모델을 학습 및 저장해주세요.")
            # Gradio 앱을 중단시키기 위해 None 반환 대신 에러 발생 또는 sys.exit() 사용도 가능
            return None
            
    def _get_transform(self):
        """이미지 전처리를 위한 Transform을 반환합니다."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image):
        """입력 이미지에 대한 예측을 수행합니다."""
        if self.model is None:
            return "모델이 로드되지 않았습니다. 서버 로그를 확인해주세요."

        # PIL 이미지를 텐서로 변환
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 각 레이블에 대한 확률을 딕셔너리로 만듦
        confidences = {LABEL_TAGS[i]: float(prob) for i, prob in enumerate(probabilities)}
        return confidences

# --- Gradio 인터페이스 생성 ---
def create_gradio_app():
    classifier = FashionClassifier()
    
    # 트렌디한 테마 설정
    theme = gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="rose",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill_dark="#111827"
    )

    with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 3rem; font-weight: 800;"> AI 패션 스타일리스트 </h1>
                <p style="font-size: 1.2rem; color: #6B7280;">이미지를 올리면 AI가 어떤 패션 아이템인지 알려드려요!</p>
            </div>
            """
        )
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="여기에 이미지를 드래그하거나 클릭하여 업로드하세요", height=300)
                submit_btn = gr.Button("결과 확인하기 ✨", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 예측 결과")
                label_output = gr.Label(num_top_classes=3, label="가장 비슷한 아이템은...", scale=1)
        
        gr.Markdown("---")
        gr.Markdown("### 🤔 이건 어떤 아이템일까요? (예시)")
        
        # 예시 이미지 추가
        gr.Examples(
            examples=[
                # 실제 경로에 예시 이미지를 넣어두면 좋습니다.
                # os.path.join(os.path.dirname(__file__), "examples/t-shirt.png"),
                # os.path.join(os.path.dirname(__file__), "examples/sneaker.jpg"),
                # os.path.join(os.path.dirname(__file__), "examples/bag.jpeg"),
            ],
            inputs=image_input,
            outputs=label_output,
            fn=classifier.predict,
            cache_examples=False,
        )

        submit_btn.click(fn=classifier.predict, inputs=image_input, outputs=label_output)

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0") 