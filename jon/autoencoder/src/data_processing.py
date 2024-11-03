from collections import namedtuple
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import copy
import torch as th
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

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
        self.data_paths = hyperparams.get('data_paths')
        self.test_data_path = hyperparams.get('test_data_path')
        self.show_week_nos = hyperparams.get('show_week_nos', False)
        
        self.merged_dfs = []
        self.XXs = []
        
        for idx, data_path in enumerate(self.data_paths):
            data = np.load(data_path)
            X = pd.DataFrame(data['X'])
            y = pd.DataFrame(data['yy'])
            
            X.rename(columns={0: "user", 1: "item", 2: "rating"}, inplace=True)
            y.rename(columns={0: "user", 1: "label"}, inplace=True)
            self.XXs.append(copy.deepcopy(X))
            
            X = X.pivot_table(index = 'user', columns='item', values = 'rating').fillna(0)
            merged_df = pd.merge(X, y, left_on = 'user', right_on = 'user', how = 'inner', validate='one_to_one')
            if self.show_week_nos:
                merged_df['week_no'] = idx
            self.merged_dfs.append(merged_df)
        
        self.merged = pd.concat(self.merged_dfs, axis = 0)
        
        self.XX = pd.concat(self.XXs, axis = 0)
        
        # Load test data
        data = np.load(self.test_data_path)
        X, y = pd.DataFrame(data['X']), pd.DataFrame(data['yy'])
        
        X.rename(columns={0: "user", 1: "item", 2: "rating"}, inplace=True)
        y.rename(columns={0: "user", 1: "label"}, inplace=True)
        X = X.pivot_table(index = 'user', columns='item', values = 'rating').fillna(0)
        
        self.test_merged = pd.merge(X, y, left_on = 'user', right_on = 'user')
        
        if self.show_week_nos:
            self.merged = pd.get_dummies(self.merged, columns=['week_no'])
        
        print("Columns in merged dataframe = ", self.merged.columns)
        print("Number of samples in class 0 = ", len(self.merged[self.merged.label == 0]))
        print("Number of samples in class 1 = ", len(self.merged[self.merged.label == 1]))
        print("Number of samples in class 2 = ", len(self.merged[self.merged.label == 2]))
        
        print("Test data samples in class 0 = ", len(self.test_merged[self.test_merged.label == 0]))
        print("Test data samples in class 1 = ", len(self.test_merged[self.test_merged.label == 1]))
        print("Test data samples in class 2 = ", len(self.test_merged[self.test_merged.label == 2]))
        
    @property
    def get_data_df(self):
        return self.merged
    
    @property
    def get_test_data_df(self):
        return self.test_merged
    
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
        assert self.test_merged is not None
        print("full dataset rows = ", self.merged.shape)
        self.new_df = UserDataProcessing.aggregate_user_ratings(self.merged, show_week_nos=self.show_week_nos)
        self.new_test_df = UserDataProcessing.aggregate_user_ratings(self.test_merged, show_week_nos=self.show_week_nos)
        self.compute_popularity = hyperparams.get('compute_popularity', False)
        if self.compute_popularity:
            self.thresholds = hyperparams.get('popularity_thresholds')
            self.initialize_popular_shows(self.merged)
        
    @staticmethod
    def aggregate_user_ratings(df, show_week_nos = False)->pd.DataFrame:
        df = copy.deepcopy(df)
        labels = df['label']
        df = df.drop('label', axis = 1)
        new_df = pd.DataFrame()
        new_df['neutral'] = (df == 0).sum(axis = 1)
        new_df['watched'] = (df == 1).sum(axis = 1)
        new_df['dislike'] = (df == -10).sum(axis = 1)
        new_df['like'] = (df == 10).sum(axis = 1)
        new_df['total'] = new_df['watched'] + new_df['like'] + new_df['dislike']
        new_df['like-prop'] = np.log(new_df['like'] + 1) - np.log(new_df['total'] + 1)
        new_df['dislike-prop'] = np.log(new_df['dislike'] + 1) - np.log(new_df['total'] + 1)
        new_df['watched-prop'] = np.log(new_df['watched'] + 1) - np.log(new_df['total'] + 1)
        new_df['like-to-dislike'] = np.log(new_df['like'] + 1) - np.log(new_df['dislike'] + 1)
        new_df = new_df.drop(['total', 'like', 'dislike', 'watched', 'neutral'], axis = 1)
        
        if show_week_nos:
            new_df['week_no_0'] = df['week_no_0'].astype(float)
            new_df['week_no_1'] = df['week_no_1'].astype(float)
            new_df['week_no_2'] = df['week_no_2'].astype(float)
            new_df['week_no_3'] = df['week_no_3'].astype(float)
            
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
    def get_test_data_df(self):
        return self.new_test_df
    
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
    
class UserDataProcessingV2(UserDataProcessing):
    def __init__(self, hyperparams: Dict):
        super().__init__(hyperparams)
        self.svd_on_majority = hyperparams.get('svd_on_majority', False)
        self.rank = hyperparams.get('rank_approx', None)
        if self.svd_on_majority:
            assert self.rank is not None
            
        assert self.new_df is not None
        self.new_df = self.add_more_user_features(self.new_df)
        self.new_test_df = self.add_more_user_features(self.new_df)
        print("New features shape = ", self.new_df.shape)
        
    def add_more_user_features(self, df)->pd.DataFrame:
        from scipy.stats import kurtosis
        df = copy.deepcopy(df)
        XX = copy.deepcopy(self.XX)
        
        labels = df['label']
        df = df.drop(['label'], axis = 'columns')
        
         # Item popularity metrics
        item_popularity = XX.groupby('item')['rating'].agg(['mean', 'count'])
        item_popularity['popularity_score'] = item_popularity['mean'] * np.log1p(item_popularity['count'])
        
        XX_with_pop = pd.merge(XX,item_popularity['popularity_score'], left_on='item', right_index =True)
        print(XX_with_pop.head(5))
        
        df['deviation_from_pop'] = XX_with_pop.groupby('user').apply(
            lambda x: np.abs(x['rating'] - x['popularity_score']).mean()
        )
        
        kurt = XX.groupby('user')['rating'].apply(lambda x: kurtosis(x))
        
        df['rating_kurtosis'] = kurt
        df['label'] = labels

        return df
        
        