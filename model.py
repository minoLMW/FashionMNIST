import torch.nn as nn

class CNN(nn.Module):
    """FashionMNIST 데이터셋을 위한 CNN 모델"""
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            # 1st conv block
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            # 2nd conv block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # 3rd conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 784),
            nn.ReLU(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x

# 레이블과 클래스명을 매핑하는 딕셔너리
LABEL_TAGS = {
    0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
} 