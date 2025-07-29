import logging
import time

def config_logging(comment=''):
    """
    로깅 설정을 구성하여 로그를 파일과 콘솔에 모두 출력합니다.
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 로그 파일 이름 설정
    if comment:
        log_filename = f'log_{comment}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    else:
        log_filename = f'log_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
        
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # 콘솔 출력 핸들러
        ]
    )

class AverageMeter(object):
    """
    주어진 기간 동안의 평균과 현재 값을 계산하고 저장합니다.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
