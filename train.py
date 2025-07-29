import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import os

from utils import AverageMeter, config_logging
from dataset import FashionMNISTCSVDataset
from model import CNN, LABEL_TAGS # model.py에서 CNN과 LABEL_TAGS를 가져옵니다.

def main():
    # argument 세팅
    parser = argparse.ArgumentParser(description='Fashion MNIST Classification')
    parser.add_argument('--batch_size', default=64, type=int, help='Dataset batch_size')
    parser.add_argument('--num_epochs', default=10, type=int, help='num epochs') # 기본 10으로 수정
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--comment', type=str, default='csv_based')

    args = parser.parse_args()
    comment = args.comment

    config_logging(comment)
    logging.info('args: {}'.format(args))

    # GPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("device: {}".format(device))

    # Transform 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 데이터셋 로드 (CSV로부터)
    train_csv_path = 'data/fashion-mnist_train.csv'
    test_csv_path = 'data/fashion-mnist_test.csv'
    
    trainset = FashionMNISTCSVDataset(path=train_csv_path, transform=transform)
    testset = FashionMNISTCSVDataset(path=test_csv_path, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    logging.info("train data length: {}, test data length: {}".format(len(trainset), len(testset)))

    # CNN 모델 인스턴스 생성 (model.py에서 가져온 클래스 사용)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 학습 루프
    total_loss = []
    for epoch in range(1, args.num_epochs + 1):
        logging.info('Train Phase, Epoch: {}'.format(epoch))
        model.train()
        train_loss = AverageMeter()

        for batch_num, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), images.shape[0])
            
            if batch_num % 100 == 0:
                logging.info(
                    "[{}/{}] # {}/{} loss: {:.4f}".format(epoch, args.num_epochs, 
                                                         batch_num, len(train_loader), train_loss.val)
                )
        total_loss.append(train_loss.avg)
        logging.info(f"Epoch {epoch} Average Loss: {train_loss.avg:.4f}")

    # 평가 루프
    def output_label(label):
        output_mapping = {
            0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 
            5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
        }
        input_val = label.item() if isinstance(label, torch.Tensor) else label
        return output_mapping[input_val]

    count = 0
    ans = 0
    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

    logging.info('Test Phase...')
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predict = torch.max(output, 1)[1]
            is_correct = (predict == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                ans += is_correct[i].item()
                count += 1
                class_correct[label] += is_correct[i].item()
                total_correct[label] += 1

    logging.info('Total Accuracy: {:.4f}%'.format((ans / count) * 100))
    for i in range(10):
        if total_correct[i] > 0:
            logging.info("Accuracy of class {}: {:.4f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))
        else:
            logging.info(f"No samples for class {output_label(i)} in test set.")

    # 학습된 모델 가중치 저장
    model_save_path = 'fashion_mnist_cnn.pth'
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"모델이 {model_save_path} 경로에 저장되었습니다.")


    # 결과 시각화
    label_tags = LABEL_TAGS # model.py에서 가져온 딕셔너리 사용
    columns = 6
    rows = 6
    fig = plt.figure(figsize=(10, 10))
    
    model.eval()
    for i in range(1, columns * rows + 1):
        data_idx = np.random.randint(len(testset))
        input_img, label_idx = testset[data_idx]
        input_img = input_img.unsqueeze(dim=0).to(device)
    
        output = model(input_img)
        _, argmax = torch.max(output, 1)
        pred = label_tags[argmax.item()]
        label = label_tags[label_idx.item()]
        
        fig.add_subplot(rows, columns, i)
        plt.title(f"{'O' if pred == label else 'X'}: {pred} ({label})")
        cmap = 'Blues' if pred == label else 'Reds'
        plot_img = input_img.squeeze().cpu().numpy()
        plt.imshow(plot_img, cmap=cmap)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
