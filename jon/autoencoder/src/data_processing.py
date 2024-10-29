from collections import namedtuple
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import copy
import torch as th

class DataFrameDataset(th.utils.data.Dataset):
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        super().__init__()
        self.df = data
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values
        features = row[:-1]
        label = row[-1]
        return th.tensor(features, dtype= th.float32), th.tensor(label, dtype=th.float32)
    
    def get_input_dim(self)->int:
        feature, _ = self.__getitem__(0)
        # print(feature, feature.shape)
        return feature.shape[0]

Popularity = namedtuple('Popularity', ['disliked_ids', 'neutral_ids', 'watched_ids', 'liked_ids'])


class DataProcessing:
    def __init__(self, hyperparams: Dict):
        self.data_path = hyperparams.get('data_path')
        data = np.load(self.data_path)
        X = pd.DataFrame(data['X'])
        y = pd.DataFrame(data['yy'])
        
        X.rename(columns={0: "user", 1: "item", 2: "rating"}, inplace=True)
        y.rename(columns={0: "user", 1: "label"}, inplace=True)
        
        X = X.pivot_table(index = 'user', columns='item', values = 'rating').fillna(0)
        self.merged = pd.merge(X, y, left_on = 'user', right_on = 'user', how = 'inner', validate='one_to_one')
        
    @property
    def get_data_df(self):
        return self.merged
    
    @property
    def get_normal_users_df(self):
        return self.merged[self.merged.label == 0]
    
    @property
    def get_anomaly_one_df(self):
        return self.merged[self.merged.label == 1]
    
    @property
    def get_anomaly_two_df(self):
        return self.merged[self.merged.label == 2]
    
    @property
    def get_anomalies_df(self):
        return self.merged[self.merged.label != 0]
    
class UserDataProcessing(DataProcessing):
    def __init__(self, hyperparams: Dict):
        super().__init__(hyperparams)
        assert self.merged is not None
        self.new_df = UserDataProcessing.aggregate_user_ratings(self.merged)
        self.compute_popularity = hyperparams.get('compute_popularity', False)
        if self.compute_popularity:
            self.thresholds = hyperparams.get('popularity_thresholds')
            self.initialize_popular_shows(self.merged)
        
    @staticmethod
    def aggregate_user_ratings(df)->pd.DataFrame:
        df = copy.deepcopy(df)
        labels = df['label']
        df = df.drop('label', axis = 1)
        new_df = pd.DataFrame()
        new_df['neutral'] = (df == 0).sum(axis = 1)
        new_df['watched'] = (df == 1).sum(axis = 1)
        new_df['dislike'] = (df == -10).sum(axis = 1)
        new_df['like'] = (df == 10).sum(axis = 1)
        new_df['total'] =new_df['watched'] + new_df['like'] + new_df['dislike']
        new_df['like-prop'] = np.log(new_df['like'] + 1) - np.log(new_df['total'] + 1)
        new_df['dislike-prop'] = np.log(new_df['dislike'] + 1) - np.log(new_df['total'] + 1)
        new_df['watched-prop'] = np.log(new_df['watched'] + 1) - np.log(new_df['total'] + 1)
        new_df['like-to-dislike'] = np.log(new_df['like'] + 1) - np.log(new_df['dislike'] + 1)
        new_df = new_df.drop(['total', 'like', 'dislike', 'watched', 'neutral'], axis = 1)
        new_df['label'] = labels
        return new_df
    
    def initialize_popular_shows(self, df)->None:
        df = copy.deepcopy(df)
        show_df = pd.DataFrame()
        df = df[df.label == 0] # don't care about anomalies
        show_df['neutral'] = (df == 0).sum(axis = 0)
        show_df['watched'] = (df == 1).sum(axis = 0)
        show_df['dislike'] = (df == -10).sum(axis = 0)
        show_df['like'] = (df == 10).sum(axis = 0)
        show_df = show_df.drop(['user', 'label'], axis = 0)
        show_df['total'] = show_df['watched'] + show_df['dislike'] + show_df['like']
        neutral_ids = show_df[(show_df.neutral / show_df.total) >= self.thresholds[0]].index.tolist()
        liked_ids = show_df[(show_df.like / show_df.total) >= self.thresholds[1]].index.tolist()
        watched_ids = show_df[(show_df.watched / show_df.total) >= self.thresholds[2]].index.tolist()
        disliked_ids = show_df[(show_df.dislike / show_df.total) >=self.thresholds[3]].index.tolist()
        
        self.popularities = Popularity(neutral_ids, liked_ids, watched_ids, disliked_ids)
        
    @property
    def get_data_df(self):
        return self.new_df
    
    @property
    def get_show_popularity(self):
        return self.popularities
    
    @property
    def get_anomaly_one_df(self):
        return self.new_df[self.new_df.label == 1]
    
    @property
    def get_anomaly_two_df(self):
        return self.new_df[self.new_df.label == 2]
    
    @property
    def get_anomalies_df(self):
        return self.new_df[self.new_df.label != 0]