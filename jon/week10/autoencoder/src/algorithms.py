from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Type
import torch as th
import torch.nn as nn
from .model import AutoEncoder
from .data_processing import DataProcessing, UserDataProcessing,DataFrameDataset
from .logs import Logger
import pandas as pd
from overrides import override
from torch.utils.data import Dataset, DataLoader
import statistics
from tqdm import tqdm

class BaseAlgorithm(ABC):
    def __init__(self, hyperparams: Dict):
        self.dataproc_class = hyperparams.get('dataproc_class')
        self.data_processing: DataProcessing = self.dataproc_class(hyperparams)
        
        self.n_epochs = hyperparams.get('n_epochs')
        self.batch_size = hyperparams.get('batch_size')
        self.optimizer_class : Callable = hyperparams.get('optimizer_class')
        self.learning_rate = hyperparams.get('learning_rate')
        # self.eval_interval = hyperparams.get('eval_interval', 1)
        self.train_dataset, self.eval_dataset = self.initialize_dataset()
        
        self.input_dim = self.train_dataset.get_input_dim()
        
        print("input_dim = ", self.input_dim)
        # Get one sample
        
        
        self.logger = Logger(hyperparams)
    
    @abstractmethod
    def train_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_model(self, epoch):
        raise NotImplementedError
    
    @abstractmethod
    def initialize_dataset(self):
        raise NotImplementedError

class AutoEncoderAlgorithm(BaseAlgorithm):
    def __init__(self, hyperparams: Dict):
        super().__init__(hyperparams)
        self.model = AutoEncoder(hyperparams, self.input_dim)
        self.optimizer : th.optim = self.optimizer_class(self.model.parameters(), lr = self.learning_rate)

    @override
    def train_model(self): 
        for epoch in tqdm(range(self.n_epochs)):
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0 # main process
            )
            # Train model
            losses = []
            for batch_idx, (features, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = th.nn.functional.mse_loss(outputs, features)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            
            self.logger.log_reconstruction_error(loss=statistics.mean(losses), epoch = epoch)
            
            # Evaluate distribution of reconstruction error
            self.evaluate_model(epoch)
            
    @override
    def evaluate_model(self, epoch):
        recon_errors = {
            0: [],
            1: [],
            2: []
        }
        eval_loader = DataLoader(
                dataset=self.eval_dataset,
                batch_size=len(self.eval_dataset),
                shuffle=True,
                num_workers=0 # main process
        )
        with th.no_grad():
            for batch_idx, (features, labels) in enumerate(eval_loader):
                # print("features.shape = ", features.shape)
                # print("labels.shape = ", labels.shape)
                outputs = self.model(features)
                # print("outputs.shape = ", outputs.shape)
                # Get item of reconstruction error
                err = th.nn.functional.mse_loss(outputs, features, reduction = 'none').sum(dim = 1)
                # print("err = ", err.shape)
                err_np = err.numpy()
                for i in range(err.shape[0]):
                    recon_errors[labels[i].item()].append(err_np[i])
                self.logger.log_recon_error_dist(recon_errors, epoch)    
    
    @override
    def initialize_dataset(self):
        data_df: pd.DataFrame = self.data_processing.get_data_df
        train_df = data_df[data_df.label == 0]
        eval_df = data_df
        
        train_dataset = DataFrameDataset(train_df)
        eval_dataset = DataFrameDataset(eval_df)
        
        return train_dataset, eval_dataset
        
