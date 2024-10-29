from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time 
class Logger:
    def __init__(self, hyperparams: Dict):
        self.logdir_path = hyperparams.get('logdir_path')
        self.writer = SummaryWriter(log_dir = f"{self.logdir_path}-{time.time_ns()//1e9}" )
        self.count = 0
        
    def flush(self):
        if (self.count > 5):
            self.writer.flush()
            self.count = 0
    
    # def log_class0_AUC(self, auc: float, epoch: int)->None:
    #     pass
    
    # def log_class1_AUC(self, auc: float, epoch: int)->None:
    #     pass
    
    # def log_class2_AUC(self, auc: float, epoch: int)->None:
    #     pass
    
    # def log_mean_AUC(self, auc0: float, auc1: float, auc2: float, epoch: int)->None:
    #     pass
    
    # def log_confusion_matrix(self, y_pred: List[int], y_true: List[int], epoch: int)->None:
    #     pass
    
    def log_reconstruction_error(self, loss: float, epoch: int)->None:
        self.writer.add_scalar(
            'Train/Reconstruction Error', loss, epoch
        )
        self.count += 1
        self.flush()
    
    def log_recon_error_dist(self, recon_errors: dict, epoch: int)->None:
        titles = {
            0: 'Normal users',
            1: 'Class 1 Anomalies',
            2: 'Class 2 Anaomalies'
        }
        for k in recon_errors:
            err = recon_errors[k]
            plt.figure()
            plt.hist(err, density=True, bins = 40)
            plt.title(f'Reconstruction Error for {titles[k]}')
            plt.xlabel('Reconstruction error')
            self.writer.add_figure(
                f'Evaluation/Class{k} Reconstruction Error',
                figure = plt.gcf(),
                global_step = epoch
            )
            self.count += len(err)
        self.flush()