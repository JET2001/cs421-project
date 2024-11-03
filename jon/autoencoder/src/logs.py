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
            
    def log_auc_wrt_epoch(self, fprs, tprs, aucs, epoch):
        plt.figure()
        colors = ['blue', 'red', 'green']
        for i in range(3):
            plt.plot(fprs[i], tprs[i], color=colors[i], lw=2, label=f'Class {i} (AUC) = {aucs[i]:0.3f}')
        plt.plot([0,1], [0,1], 'k--', lw = 2)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Multiclass Classification")
        plt.legend(loc= "lower right")
        plt.grid(True)
        self.writer.add_figure(
            "Evaluation/AUC against epochs",
            plt.gcf(),
            epoch
        )
        
        self.writer.add_scalar(
            'Evaluation/AUC0 against epoch', aucs[0], epoch
        )
        self.writer.add_scalar(
            'Evaluation/AUC1 against epoch', aucs[1], epoch
        )
        self.writer.add_scalar(
            'Evaluation/AUC2 against epoch', aucs[2], epoch
        )
        
        self.count += 5
        self.flush()
        
    def log_reconstruction_error(self, loss: float, epoch: int, class_no: int)->None:
        self.writer.add_scalar(
            f'Train/Reconstruction Error Class{class_no}', loss, epoch
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