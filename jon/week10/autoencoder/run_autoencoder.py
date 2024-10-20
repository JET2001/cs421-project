import torch as th
from src.algorithms import BaseAlgorithm, AutoEncoderAlgorithm
from src.data_processing import DataProcessing, UserDataProcessing

LATENT_SPACE = 1
hyperparams = {
    'n_epochs': 100,
    'batch_size': 32,
    'optimizer_class': th.optim.Adam,
    'dataproc_class': UserDataProcessing,
    'data_path': './data/second_batch_multi_labels.npz',
    'learning_rate': 1e-3,
    'compute_popularity': False,
    'popularity_thresholds': None,
    'hidden': [10, 5, 3],
    'latent_space_dim': LATENT_SPACE,
    'logdir_path': f'./runs/autoenc-latent-EDA-With-LikeOdds-{LATENT_SPACE}'
}

if __name__ == '__main__':
    autoenc = AutoEncoderAlgorithm(hyperparams)
    autoenc.train_model()