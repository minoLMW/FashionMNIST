import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FashionMNISTCSVDataset(Dataset):
    """FashionMNIST CSV 파일을 로드하는 커스텀 데이터셋"""
    
    def __init__(self, path, transform=None):
        """
        Args:
            path (string): CSV 파일 경로
            transform (callable, optional): 샘플에 적용될 Transform
        """
        self.data = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 첫 번째 열은 레이블, 나머지는 이미지 픽셀
        label = self.data.iloc[idx, 0]
        # 28x28 이미지로 재구성하고, 0-255 범위의 uint8 타입으로 변경
        image = self.data.iloc[idx, 1:].values.astype('uint8').reshape((28, 28))
        
        sample = {'image': image, 'label': label}

        if self.transform:
            # transform은 PIL 이미지를 기대하는 경우가 많으므로, PIL 이미지로 변환
            from PIL import Image
            image = Image.fromarray(sample['image'], mode='L')
            sample['image'] = self.transform(image)
        
        sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
            
        return sample['image'], sample['label']
