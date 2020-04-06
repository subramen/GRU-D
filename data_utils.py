# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************

import pandas as pd
import utils
import pickle
import numpy as np
import torch.utils.data as data
import random, os, sys
from sklearn.preprocessing import MinMaxScaler


class TargetFunc:
    def __init__(self, data):
        self.data = data

    def tgt1(self, indices, thresholds=[2, 3]): # 3 target classes
        y = self.data.loc[indices[0]:indices[1]].col5.sum() 
        thresholds.append(sys.maxsize)
        for c,t in enumerate(thresholds):
            if y<=t:
                return c

    def tgt2(self, indices):
        # Mark 1 if even one occurrence, else 0
        df = self.data.loc[indices[0]:indices[1]]
        return min(1, df.col5.fillna(0).sum())
    
    
     
class SequenceGenerator:
    def __init__(self, history_len, future_len, col_dict):
        """
        Patient Sequence-Label generator
        history_len: length of training sequence (default=24 months)
        future_len: length of future sequence to predict on (default=8 months)
        """
        self.history_len = history_len
        self.future_len = future_len
        self.min_rows_reqd=self.history_len+self.future_len
        self.col_dict=col_dict

   
    def df_to_training_pairs(self, seq_dir, auto_id, grp):
       
        def decompose_slice(indices):            
            input_df = grp.loc[indices[0]:indices[1]] # Slice out an input sequence
            first_month = input_df.MONTH.iloc[0]
            seq_name = f'{auto_id}_{first_month}'
            
            # input_df = pd.DataFrame(self.scaler.transform(input_df), columns=input_df.columns)
            x = input_df[self.col_dict['input']] 
            x_obs = x.shift(1).fillna(x.iloc[0]).values
            x = x.values
            m = input_df[self.col_dict['missing']].values
            t = input_df[self.col_dict['delta']].values
            
            return seq_name, np.array([x, m, t, x_obs]) 

        input_start_ix = grp.index[0]
        input_final_ix = grp.index[-1]-self.min_rows_reqd+1
        input_seq_df = [(pos, pos + self.history_len-1) for pos in range(input_start_ix, input_final_ix+1)] # pandas slicing is inclusive

        target_start_ix = grp.index[0] + self.history_len
        target_final_ix = grp.index[-1]-self.future_len+1
        target_seq_df = [(pos, pos+self.future_len-1) for pos in range(target_start_ix, target_final_ix+1)]
    
        tgt_types = ['tgt1', 'tgt2']
        label_dict={}
        target_loader = TargetFunc(grp)

        # sliding window
        for slice_ix in range(len(input_seq_df)):
            name, x = decompose_slice(input_seq_df[slice_ix])
            utils.pkl_dump(x, os.path.join(seq_dir, name+'.npy'))
    
            label_dict[name] = {}
            for tgt in tgt_types:
                fn = eval(f"target_loader.{tgt}")
                label_dict[name][tgt] = fn(target_seq_df[slice_ix])
        
        return label_dict # return for collation with other patients

  
    def train_test_split(self, dataset, x_mean, datadir, train_size):
        """
        IBDSequenceGenerator.write_to_disk:
         - Reads padded X, M, T matrices
         - Generate overlapping sequences and {tgt_type} training labels for each patient
         - Persist sequences as numpy arrays at data/{datadir}/{train, valid, test}/{ID}_{START_TS}.npy
         - Persist labels for all sequences in data/{datadir}/{train, valid, test}/label_dict.pkl
         - Persist empirical mean values at data/{datadir}/x_mean.pkl
        """
        print(SequenceGenerator.write_to_disk.__doc__)

        # Create out directories
        if not datadir: datadir=f'm{self.history_len}p{self.future_len}'
        out_fp = os.path.join('data', datadir)
        train_seq_dir = os.path.join(out_fp, 'train')
        valid_seq_dir = os.path.join(out_fp, 'valid')
        test_seq_dir = os.path.join(out_fp, 'test')
        if not os.path.exists(out_fp): os.makedirs(out_fp)
        if not os.path.exists(train_seq_dir): os.makedirs(train_seq_dir)
        if not os.path.exists(valid_seq_dir): os.makedirs(valid_seq_dir)
        if not os.path.exists(test_seq_dir): os.makedirs(test_seq_dir)
                    
        groups = dataset.groupby('ID')

        TRAIN_LABELS = {}
        VALID_LABELS = {}
        TEST_LABELS = {}
        valid_size = (1+train_size)/2
        patients_dropped = 0
        
        for auto_id, grp in groups:
            grplen = grp.shape[0]
            
            if grplen < self.min_rows_reqd: 
                patients_dropped+=1
                continue

            train = grp.iloc[:int(train_size*grplen)]
            valid = grp.iloc[int(train_size*grplen):int(valid_size*grplen)]
            test = grp.iloc[int(valid_size*grplen):]

            # if not enough observations for a validation sequence, use them either as test or train (don't waste any data)
            if valid.shape[0] < self.min_rows_reqd: 
                if valid.shape[0]+test.shape[0] >= self.min_rows_reqd: 
                    test = pd.concat([valid,test])
                    valid = None
                else: 
                    train = pd.concat([train, valid, test])
                    valid = None
                    test = None
            
            TRAIN_LABELS.update(self.df_to_training_pairs(train_seq_dir, auto_id, train))
            if valid is not None:
                VALID_LABELS.update(self.df_to_training_pairs(valid_seq_dir, auto_id, valid))
            if test is not None:
                TEST_LABELS.update(self.df_to_training_pairs(test_seq_dir, auto_id, test))

        utils.pkl_dump(TRAIN_LABELS, os.path.join(train_seq_dir, 'label_dict.pkl'))
        utils.pkl_dump(VALID_LABELS, os.path.join(valid_seq_dir, 'label_dict.pkl'))
        utils.pkl_dump(TEST_LABELS, os.path.join(test_seq_dir, 'label_dict.pkl'))
        
        utils.pkl_dump(x_mean[2:], os.path.join(out_fp, 'x_mean.pkl')) # remove auto id and month ts
        print(f"Patients dropped: {patients_dropped}")



class SeqDataset(data.Dataset):  
    def __init__(self, datadir, type): 
        self.datadir = os.path.join('data',datadir, type) 
        self.label_dict = utils.pkl_load(os.path.join(self.datadir,'label_dict.pkl'))
        self.list_IDs = list(self.label_dict.keys())

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = utils.pkl_load(os.path.join(self.datadir, ID+'.npy')) # Numpy seq
        y = self.label_dict[ID] # Dict of available target values
        return X, y



def clean_longitudinal_inputs(df):

    # HELPER FUNCTIONS
    def apply_monthly_timestamp(df): 
        """
        Applying a monthly timestamp to each observation. Each patient's first observation starts with 0. 
        Subsequent observations have a timestamp denoting the number of months since first observation
        """
        print(apply_monthly_timestamp.__doc__)

        df['tmp_ts'] = (df.YEAR.astype(str)+df.MONTH.astype(str).str.zfill(2)).astype(int)
        t0_indices = df.groupby('ID').tmp_ts.idxmin()     # find earliest row
        t0_df = df.loc[t0_indices][['ID','YEAR','MONTH']]
        t0_df.columns = ['ID', 'tmp_minY', 'tmp_minMo']
        # add tmp columns to df for vectorized calculation of delta in months
        timestamped_df = df.merge(t0_df, how='left', on='ID')
        timestamped_df['MONTH'] = (timestamped_df['YEAR']-timestamped_df['tmp_minY'])*12 + timestamped_df['MONTH']-timestamped_df['tmp_minMo']
        df = timestamped_df[[x for x in timestamped_df.columns if x[:3]!='tmp']]
        print(f"Data shape: {df.shape}")
        return df
    

    def pad_missing_months(timestamped_df):
        """
        Resampling the data to a monthly rate.
        """
        print(pad_missing_months.__doc__)

        def pad(df):
            df = df.reset_index()
            new_index = pd.MultiIndex.from_product([[df.ID.values[0]], list(range(0, df.MONTH.max()+1))], names=['ID', 'MONTH'])
            df = df.set_index(['ID', 'MONTH']).reindex(new_index).reset_index()
            return df

        df = timestamped_df.groupby('ID', as_index=False).apply(pad)
        df.index = range(0, df.shape[0])
        df.drop(['index', 'YEAR', 'MONTH'], axis=1, inplace=True)
        print(f"Data shape: {df.shape}")
        return df

    def missingness_indicators(padded):
        """
        Computing the `missing_mask` and `missing_delta` matrices. 
        missing_mask is 1 if variable is observed, 0 if not.
        missing_delta tracks the number of `time_interval`s (here, 1 month) between two observations of a variable
        """
        print(missingness_indicators.__doc__)

        missing_mask = padded.copy()
        missing_mask.loc[:] = missing_mask.loc[:].notnull().astype(int)
        missing_mask['ID'] = padded['ID'].copy() # restore auto_ids
        missing_delta = missing_mask.copy()
        
        time_interval = 1 
        missing_delta['tmp_delta'] = time_interval
        for observed in missing_delta.columns[1:]:
            missing_delta['obs_delays'] = missing_delta.groupby('ID')[observed].cumsum()
            missing_delta[observed] = missing_delta.groupby(['ID', 'obs_delays'])['tmp_delta'].cumsum() -1
        missing_delta = missing_delta.drop(['tmp_delta', 'obs_delays'], axis=1)

        missing_mask.columns = ['ID']+[f'MISSING_{x}' for x in padded.columns[1:]]
        missing_delta.columns = ['ID']+[f'DELTA_{x}' for x in  padded.columns[1:]]
        return missing_mask.drop(['ID'],axis=1), missing_delta.drop(['ID'],axis=1)

    
    df = apply_monthly_timestamp(df)
    x_mean = df.mean()
    df = pad_missing_months(df)
    m, t = missingness_indicators(df)
    df = df.fillna(None, 'ffill')
    print("Imputation done")
    
    all_data = pd.concat([df, m, t], axis=1)
    col_dict = {'input':df.columns, 'missing':m.columns, 'delta':t.columns}
    
    return all_data, col_dict, x_mean


if __name__ == "__main__":
    
    # Generate dummy data
    arr = pd.DataFrame(np.random.rand(5,5,5).reshape(25,5), columns=[f'col{i+1}' for i in range(5)])
    arr['ID'] = sum([[i]*5 for i in range(3,8)], [])
    arr['YEAR'] = 2020
    arr['MONTH'] = [1,4,5,7,8]*5
 
    # Generate cleaned, resampled, imputed data with missing mask and delta + save to pickles folder
    df, col_dict, x_mean = clean_longitudinal_inputs(arr) 

    # Init SequenceGenerator with look-back and look-forward durations
    seqgen = SequenceGenerator(3,1, col_dict)

    # Generate time series inputs with their target labels
    # X (input arrays) pickled to disk at /tmp/input_id.npy 
    # y returned as a dict {input_id : {tgt1: 0, tgt2: 1}}
    c=0
    target_dict = {}
    for auto_id, group in df.groupby('ID'):
        target_dict.update(seqgen.df_to_training_pairs('tmp', auto_id, group))
    
    utils.pkl_dump(target_dict, 'tmp/label_dict.pkl')


