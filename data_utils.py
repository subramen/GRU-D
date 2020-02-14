# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************

import pandas as pd
import utils
import pickle
import numpy as np
import torch.utils.data as data
import random, os



def pad_missing_months(timestamped_df):
    
    cols = timestamped_df.columns

    def pad(df): # Adds nan rows between non-consecutive discrete timesteps
        df = df.reset_index()
        new_index = pd.MultiIndex.from_product([[df.ID.values[0]], list(range(0, df.TS.max()+1))], names=['ID', 'TS'])
        df = df.set_index(['ID', 'TS']).reindex(new_index)#.reset_index()
        return df
    
    padded = timestamped_df.groupby('ID', as_index=False).apply(pad).reset_index()[cols]
    return padded


def missingness_indicators(padded, prefixes=['MISSING', 'DELTA']):
    
    missing_mask = padded.copy()
    missing_mask.loc[:] = missing_mask.loc[:].notnull().astype(int)
    missing_delta = missing_mask.copy()

    time_step = 1 
    missing_delta['tmp_delta'] = time_step
    for observed in missing_delta.columns:
        missing_delta['tmp_obs_delays'] = missing_delta.groupby('ID')[observed].cumsum() # Unchanging when missing,increaseing when observed
        missing_delta[observed] = missing_delta.groupby(['ID', 'tmp_obs_delays'])['tmp_delta'].cumsum() # Counts timesteps since last observed
    missing_delta = missing_delta.drop(['tmp_delta', 'tmp_obs_delays'], axis=1)

    missing_mask.columns = [f'{prefixes[0]}_{x}' for x in missing_mask.columns]
    missing_delta.columns = [f'{prefixes[1]}_{x}' for x in  missing_delta.columns]
    return missing_mask, missing_delta  


def x_last_observed(x):
    df = x.shift(1).fillna(x.iloc[0])
    df.columns = [f'LO_{c}' for c in x.columns]
    return df


    df = pad_missing_months_(df)
    m, t = missingness_indicators(df)

    all_data = pd.concat([df, m, t, y], axis=1)


class Imputer:
    
    def mean_imputer(self, df):
        obs_mean = df.loc[df.PADDED==0, self.cols_to_impute].mean()
        return df.fillna(obs_mean)

    def forward(self, grp, default=0):
        # fillna of 0th row with default values    
        grp.iloc[0] = grp.iloc[0].fillna(default)
        grp.loc[:] = grp.loc[:].fillna(None, 'ffill')
        return grp

    def grp_forward(self, df, groupby_cols, default_values):
        return df.groupby(groupby_cols).apply(self.forward, default_values=default_values)

    def get(self, how):
        out=None
        if how=='grp_forward':
            return self.grp_forward
        if how=='forward-mini':
            return self.forward_mini
        if how=='mean':
            return self.mean_imputer

