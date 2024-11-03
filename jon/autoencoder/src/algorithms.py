from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Type
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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures, label_binarize, MinMaxScaler, StandardScaler
import numpy as np
import time
import copy
import statistics

class BaseAlgorithm(ABC):
    def __init__(self, hyperparams: Dict):
        self.dataproc_class = hyperparams.get('dataproc_class')
        self.data_processing: DataProcessing = self.dataproc_class(hyperparams)
        
        self.n_epochs = hyperparams.get('n_epochs')
        self.batch_sizes = hyperparams.get('batch_sizes')
        self.optimizer_class : Callable = hyperparams.get('optimizer_class')
        self.learning_rate = hyperparams.get('learning_rate')
        self.standardize_features = hyperparams.get('standardize_features', False)
        self.poly_features = hyperparams.get('poly_features', False)
        self.poly_degree = hyperparams.get('poly_degree', None)
        if self.poly_features:
            assert self.poly_degree is not None
        # self.eval_interval = hyperparams.get('eval_interval', 1)
        self.train_datasets, self.eval_datasets = self.initialize_dataset()
        
        self.input_dim = self.train_datasets[0].get_input_dim()
        print("input_dim = ", self.input_dim)
        self.time_str = time.time_ns()//1e9
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
        self.hidden_layers = hyperparams.get('hidden_layers')
        self.latent_space_dims = hyperparams.get('latent_space_dims')
        
        self.models : List[nn.Module] = [AutoEncoder(self.hidden_layers[i], self.latent_space_dims[i], self.input_dim) for i in range(3)]
        self.optimizers : List[th.optim] = [self.optimizer_class(self.models[i].parameters(), lr = self.learning_rate) for i in range(3)]
        
        self.scaled_recon_errors = hyperparams.get('scaled_recon_errors', False)
        self.score_calc_avg = hyperparams.get('score_calc_avg', False)
        
        self.best_models = {i: self.models[i].state_dict() for i in range(3)}
        self.best_models['epoch'] = -1
        self.best_auc = 0
        self.max_epochs = hyperparams.get('max_epochs')
        
        self.scalers: List[StandardScaler] = [StandardScaler() for _ in range(3)]
        self.scaled_errors_const = hyperparams.get('scaled_errors_const', False)
        
        
    @override
    def train_model(self): 
        
        for epoch in tqdm(range(self.n_epochs)):
            for anomaly_class in [0,1,2]: 
                if epoch > self.max_epochs[anomaly_class]:
                    continue
                if epoch == self.max_epochs[anomaly_class]:
                    # Load the best model, we will no longer touch this
                    self.models[anomaly_class].load_state_dict(self.best_models[anomaly_class])
                    continue
                train_loader = DataLoader(
                    dataset=self.train_datasets[anomaly_class],
                    batch_size=self.batch_sizes[anomaly_class],
                    shuffle=True,
                    num_workers=0 # main process
                )
                # Train model
                losses = []
                for batch_idx, (features, labels) in enumerate(train_loader):
                    self.optimizers[anomaly_class].zero_grad()
                    outputs = self.models[anomaly_class](features)
                    loss = th.nn.functional.mse_loss(outputs, features)
                    loss.backward()
                    self.optimizers[anomaly_class].step()
                    losses.append(loss.item())
                    err = th.nn.functional.mse_loss(outputs, features, reduction = 'none').sum(dim = 1)
                    err_np = th.clone(err)
                    err_np = err_np.detach().numpy()
                    
                    if self.scaled_recon_errors:
                        self.scalers[anomaly_class] = self.scalers[anomaly_class].fit(err_np.reshape(-1,1))
                    
                    
                
                self.logger.log_reconstruction_error(loss=statistics.mean(losses), epoch = epoch, class_no = anomaly_class)
                
            # Evaluate distribution of reconstruction error
            self.evaluate_model(epoch)
        
        for model_state_dict in [0,1,2]:
            th.save(self.best_models[model_state_dict],  f'models/Class{model_state_dict}/class{model_state_dict}-epoch-{self.best_models['epoch']}-auc-{self.best_auc}-{self.time_str}.pth')
        
            
    @override
    def evaluate_model(self, epoch):
        recon_errors = {
            0: [],
            1: [],
            2: []
        }
        recon_tensors = []
        for anomaly_class in [0, 1, 2]:
            if epoch >= self.max_epochs[anomaly_class]:
                self.models[anomaly_class].load_state_dict(self.best_models[anomaly_class])
            eval_loader = DataLoader(
                    dataset=self.eval_datasets[anomaly_class],
                    batch_size=len(self.eval_datasets[anomaly_class]),
                    shuffle=False,
                    num_workers=0 # main process
            )
            with th.no_grad():
                for batch_idx, (features, labels) in enumerate(eval_loader):
                    outputs = self.models[anomaly_class](features)
                    err = th.nn.functional.mse_loss(outputs, features, reduction = 'none').sum(dim = 1)
                    err_np = err.numpy()
                    for i in range(err.shape[0]):
                        recon_errors[labels[i].item()].append(err_np[i])
                    
                    self.logger.log_recon_error_dist(recon_errors, epoch) 
                    
                if self.scaled_recon_errors:
                    # scaler = MinMaxScaler(feature_range=(0, 5)).fit(err_np.reshape(1, -1))
                    # err_scaled = scaler.transform(err_np.reshape(1, -1))
                    # recon_tensors.append(th.tensor(err_scaled).reshape(err_scaled.shape[1], 1))
                    err_scaled = self.scalers[anomaly_class].transform(err_np.reshape(-1,1)) 
                    if self.scaled_errors_const:
                        err_scaled += 3.
                    recon_tensors.append(th.tensor(err_scaled).reshape(err_scaled.shape[0], 1))
                else:       
                    recon_tensors.append(err)
        scores = th.column_stack(recon_tensors)
        
        if self.score_calc_avg:
            probabilities = self.simple_average_for_probs(scores)
        else:
            probabilities = self.sample_softmax(scores)
        
        assert np.isclose(np.sum(probabilities[0, :]), 1.0), f"Probabilities = {probabilities}"
        test_labels = np.array(labels)
        y_true = label_binarize(test_labels, classes=[0,1,2])
        
        fprs, tprs, roc_aucs = dict(), dict(), dict()
        for i in range(3):
            fprs[i], tprs[i], _ = roc_curve(y_true[:, i], probabilities[:, i])
            roc_aucs[i] = auc(fprs[i], tprs[i])
            # Save best combination of model
        if statistics.mean([roc_aucs[0], roc_aucs[1], roc_aucs[2]]) > self.best_auc:
            self.best_models[0] = self.models[0].state_dict()
            self.best_models[1] = self.models[1].state_dict()
            self.best_models[2] = self.models[2].state_dict()
            self.best_models['epoch'] = epoch
            self.best_auc = statistics.mean([roc_aucs[0], roc_aucs[1], roc_aucs[2]]) 
        
        self.logger.log_auc_wrt_epoch(
            fprs,
            tprs,
            roc_aucs,
            epoch
        )
            
        
        
    def sample_softmax(self, scores)->np.array:
        scores = -1*scores 
        # print("scores = ", scores, scores.shape)
        # note that max here corresponds to the minimum score in the original 
        # score array
        shifted_scores = scores - th.max(scores, axis = 1, keepdim=True).values
        # print("shifted_scores = ", shifted_scores, shifted_scores.shape)
        exp_scores = th.exp(shifted_scores).numpy()
        probabilities = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
        return probabilities
    
    def simple_average_for_probs(self, scores)->np.array:
        print("scores = ", scores)
        scores = 1.0 / scores
        
        score_over_row = th.sum(scores, axis = 1, keepdim=True)
        # print("score_over_row = ", score_over_row)
        probabilities = (scores / score_over_row)
        return probabilities.numpy()
    
    @override
    def initialize_dataset(self):
        train_df: pd.DataFrame = self.data_processing.get_data_df
        test_df : pd.DataFrame = self.data_processing.get_test_data_df
        train_dfs = []
        eval_dfs = []
        
        if self.standardize_features:
            for anomaly_class in [0,1,2]:
                train_data = copy.deepcopy(train_df[train_df.label == anomaly_class])
                eval_data = copy.deepcopy(test_df)
                eval_labels = test_df['label']
                train_labels = train_data['label']
                train_data = train_data.drop('label', axis = 'columns')
                eval_data = eval_data.drop('label', axis = 'columns')
                
                sr = StandardScaler().fit(train_data)
                train_data_scaled = sr.transform(train_data)
                eval_data_scaled = sr.transform(eval_data)
                
                if self.poly_features:
                    assert self.poly_degree is not None
                    poly = PolynomialFeatures(degree=self.poly_degree).fit(train_data_scaled)
                    train_data_scaled = poly.transform(train_data_scaled)
                    eval_data_scaled = poly.transform(eval_data_scaled)
                    
                    sr2 = StandardScaler().fit(train_data_scaled)
                    train_data_scaled = sr2.transform(train_data_scaled)
                    eval_data_scaled = sr2.transform(eval_data_scaled)
                
                train_data_scaled = pd.DataFrame(train_data_scaled)
                eval_data_scaled = pd.DataFrame(eval_data_scaled)
                assert len(train_data_scaled) == len(train_labels)
                assert len(eval_data_scaled) == len(eval_labels)
                train_data_scaled['label'] = train_labels.tolist()
                eval_data_scaled['label'] = eval_labels.tolist()
                train_dfs.append(train_data_scaled)
                eval_dfs.append(eval_data_scaled)
        else:        
            train_dfs.append(train_df[train_df.label == 0])
            train_dfs.append(train_df[train_df.label == 1])
            train_dfs.append(train_df[train_df.label == 2])
            
            eval_dfs.append(test_df)
            eval_dfs.append(test_df)
            eval_dfs.append(test_df)
        
        train_datasets = [DataFrameDataset(x) for x in train_dfs]
        eval_datasets = [DataFrameDataset(x) for x in eval_dfs]
        
        return train_datasets, eval_datasets
        
