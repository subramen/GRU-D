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

def clean_longitudinal_inputs(load_pkl=True, padded=True, exclude_ct=True):
    if load_pkl:
        df = pd.read_pickle('pickles/monthwise_inputs.pkl')
        col_dict = utils.pkl_load('pickles/col_dict.pkl')
        x_mean = utils.pkl_load('pickles/x_mean.pkl')
        return df, col_dict, x_mean

    # HELPER FUNCTIONS

    def apply_monthly_timestamp(df): 
        """
        Applying a monthly timestamp to each observation. Each patient's first observation starts with 0. 
        Subsequent observations have a timestamp denoting the number of months since first observation
        """
        print(apply_monthly_timestamp.__doc__)

        df['tmp_ts'] = (df.RESULT_YEAR.astype(str)+df.RESULT_MONTH.astype(str).str.zfill(2)).astype(int)
        t0_indices = df.groupby('AUTO_ID').tmp_ts.idxmin()     # find earliest row
        t0_df = df.loc[t0_indices][['AUTO_ID','RESULT_YEAR','RESULT_MONTH']]
        t0_df.columns = ['AUTO_ID', 'tmp_minY', 'tmp_minMo']
        # add tmp columns to df for vectorized calculation of delta in months
        timestamped_df = df.merge(t0_df, how='left', on='AUTO_ID')
        timestamped_df['MONTH_TS'] = (timestamped_df['RESULT_YEAR']-timestamped_df['tmp_minY'])*12 + timestamped_df['RESULT_MONTH']-timestamped_df['tmp_minMo']
        df = timestamped_df[[x for x in timestamped_df.columns if x[:3]!='tmp']]
        print(f"Data shape: {df.shape}")
        return df
    
    def put_empirical_mean(df):
        x_mean = Imputer(df).commonsense_imputer().mean()
        utils.pkl_dump(MinMaxScaler((-0.5, 0.5)).fit_transform(x_mean.values.reshape(1,-1)), 'pickles/x_mean.pkl')
        print("Empirical mean pickled at 'pickles/x_mean.pkl'")

    def pad_missing_months(timestamped_df):
        """
        Resampling the data to a monthly rate.
        """
        print(pad_missing_months.__doc__)

        def pad(df):
            df = df.reset_index()
            new_index = pd.MultiIndex.from_product([[df.AUTO_ID.values[0]], list(range(0, df.MONTH_TS.max()+1))], names=['AUTO_ID', 'MONTH_TS'])
            df = df.set_index(['AUTO_ID', 'MONTH_TS']).reindex(new_index).reset_index()
            return df
        df = timestamped_df.groupby('AUTO_ID', as_index=False).apply(pad)
        df.index = range(0, df.shape[0])
        df.drop(['index', 'RESULT_YEAR', 'RESULT_MONTH'], axis=1, inplace=True)
        print(f"Data shape: {df.shape}")
        return df

    def put_scaler(df):
        """
        Fit scaler to normalize variables in a range of -0.5 to +0.5
        """
        print(put_scaler.__doc__)
        scaler = MinMaxScaler((-0.5, 0.5)).fit(df)
        utils.pkl_dump(scaler, 'pickles/input_scaler.pkl')
        print("Saved fitted scaler to pickles/input_scaler.pkl")
        # return scaler

    def missingness_indicators(padded):
        """
        Computing the `missing_mask` and `missing_delta` matrices. 
        missing_mask is 1 if variable is observed, 0 if not.
        missing_delta tracks the number of `time_interval`s (here, 1 month) between two observations of a variable
        """
        print(missingness_indicators.__doc__)

        missing_mask = padded.copy()
        missing_mask.loc[:] = missing_mask.loc[:].notnull().astype(int)
        missing_mask['AUTO_ID'] = padded['AUTO_ID'].copy() # restore auto_ids
        missing_delta = missing_mask.copy()
        
        time_interval = 1 
        missing_delta['tmp_delta'] = time_interval
        for observed in missing_delta.columns[1:]:
            missing_delta['obs_delays'] = missing_delta.groupby('AUTO_ID')[observed].cumsum()
            missing_delta[observed] = missing_delta.groupby(['AUTO_ID', 'obs_delays'])['tmp_delta'].cumsum() -1
        missing_delta = missing_delta.drop(['tmp_delta', 'obs_delays'], axis=1)

        missing_mask.columns = ['AUTO_ID']+[f'MISSING_{x}' for x in padded.columns[1:]]
        missing_delta.columns = ['AUTO_ID']+[f'DELTA_{x}' for x in  padded.columns[1:]]
        return missing_mask.drop(['AUTO_ID'],axis=1), missing_delta.drop(['AUTO_ID'],axis=1)


    df = pd.read_csv('/time/series/events.csv')
    df = apply_monthly_timestamp(df)
    put_empirical_mean(df)
    df = pad_missing_months(df)
    m, t = missingness_indicators(df)
    put_scaler(pd.concat([df,m,t], axis=1))
    df = Imputer(df).clinical_impute() # Impute acc to clinical rules
    print("Imputation done")
    
    # Pickle everything
    try: os.makedirs('pickles')
    except FileExistsError: pass

    df.to_pickle('pickles/x_padded_inputs.pkl')
    m.to_pickle('pickles/m_missing_mask.pkl')
    t.to_pickle('pickles/t_missing_delta.pkl')
    all_data = pd.concat([df, m, t], axis=1)
    all_data.to_pickle('pickles/monthwise_inputs.pkl')  
    col_dict = {'input':df.columns, 'missing':m.columns, 'delta':t.columns}
    with open('pickles/col_dict.pkl', 'wb') as f:
        pickle.dump(col_dict, f)
    
    print("All data saved in pickles/")


class Imputer:
    def __init__(self, df=None):
        self.df = df

        self.cols_to_impute = ['ENC_OFF_Related', 'ENC_OFF_Unrelated',
       'ENC_PROC_Related', 'ENC_PROC_Unrelated', 'ENC_TEL_Related',
       'ENC_TEL_Unrelated', 'ENC_SURGERY', 'DIAG_COLONOSCOPY',
       'DIAG_ENDOSCOPY', 'DIAG_SIGMOIDOSCOPY', 'DIAG_ILEOSCOPY', 'DIAG_ANO',
       'DIAG_CT_ABPEL', 'LAB_albumin_High', 'LAB_albumin_Low',
       'LAB_albumin_Normal', 'LAB_crp_High', 'LAB_crp_Low', 'LAB_crp_Normal',
       'LAB_eos_High', 'LAB_eos_Low', 'LAB_eos_Normal', 'LAB_esr_High',
       'LAB_esr_Normal', 'LAB_hemoglobin_High', 'LAB_hemoglobin_Low',
       'LAB_hemoglobin_Normal', 'LAB_monocytes_High', 'LAB_monocytes_Low',
       'LAB_monocytes_Normal', 'LAB_vitamin_d_High', 'LAB_vitamin_d_Low',
       'LAB_vitamin_d_Normal', 'MED_5_ASA', 'MED_Systemic_steroids',
       'MED_Immunomodulators', 'MED_Psych', 'MED_Vitamin_D', 'MED_ANTI_TNF',
       'MED_ANTI_IL12', 'MED_ANTI_INTEGRIN', 'HBI_CROHNS_SCORE',
       'HBI_UC_SCORE', 'DS_CD', 'DS_UC', 'DS_AGE_DX', 'DS_PREV_RESECTION']

        self.enc_diag_rx_cols = [x for x in self.cols_to_impute if x[:3]=='ENC' or x[:4]=='DIAG' or x[:3]=='MED']
        self.hbi_cols = [x for x in self.cols_to_impute if x[:3]=='HBI']
        self.lab_fillna = {f'LAB_{g}_{l}':(1 if l=='Normal' else 0) \
                            for g in ['albumin', 'eos', 'hemoglobin', 'monocytes', 'vitamin_d', 'crp', 'esr'] \
                                for l in ['High', 'Low', 'Normal']}
    
    def clinical_impute(self):
        """
        Clinical Imputation:
         - Use ffill for LABS, HBI Scores
         - Use commonsense for DIAG, ENC,  MEDS (if it isn't recorded, it didn't happen -> impute 0)
        """
        print(Imputer.clinical_impute.__doc__)
        df = self.df.copy()
        df.loc[:, self.enc_diag_rx_cols] = df.loc[:, self.enc_diag_rx_cols].fillna(0) # commonsense
        df = df.groupby('AUTO_ID').apply(self.forward) # ffill
        return df

    def forward(self, grp):
        # fillna of 0th row with commonsense values    
        grp.loc[grp.MONTH_TS==0,:] = self.commonsense_imputer(grp.loc[grp.MONTH_TS==0,:])
        grp.loc[:,self.cols_to_impute] = grp.loc[:,self.cols_to_impute].fillna(None, 'ffill')
        return grp.fillna(0)

    def commonsense_imputer(self, df=None):
        """
        - 0 for encounters (if it wasn't recorded, it probably didn't happen)
        - 0 for diagnoses (-- " --)
        - 0 for RX (-- " --)
        - 1 for LAB_*_Normal (if it wasn't prescribed, it's probably normal)
        - -1 for HBI (wasn't conducted)
        """
        if df is None:
            df = self.df.copy()
        # impute encounter, diag, meds
        df.loc[:, self.enc_diag_rx_cols] = df.loc[:, self.enc_diag_rx_cols].fillna(0)
        # impute labs
        df = df.fillna(self.lab_fillna)
        # impute HBI
        df.loc[:,self.hbi_cols] = df.loc[:,self.hbi_cols].fillna(-1)
        return df

    
    


class TargetFunc:
    def __init__(self, data):
        self.data = data

    def tgt1(self, indices, thresholds=[10000, 80000]): # 3 target classes
        y = self.data.loc[indices[0]:indices[1]].tgt_col1.sum() 
        thresholds.append(sys.maxsize)
        for c,t in enumerate(thresholds):
            if y<=t:
                return c

    def tgt2(self, indices):
        # Mark 1 if even one occurrence, else 0
        df = self.data.loc[indices[0]:indices[1]]
        return min(1, df.tgt_col2.fillna(0).sum())
    
    
   
        
class IBDSequenceGenerator:
    def __init__(self, history_len, future_len):
        """
        Patient Sequence-Label generator
        history_len: length of training sequence (default=24 months)
        future_len: length of future sequence to predict on (default=8 months)
        """
        self.history_len = history_len
        self.future_len = future_len
        self.min_rows_reqd=self.history_len+self.future_len
        self.scaler = utils.pkl_load('pickles/input_scaler.pkl')
        self.col_dict=None

   
    def df_to_training_pairs(self, seq_dir, auto_id, grp):
       
        def decompose_slice(indices):            
            input_df = grp.loc[indices[0]:indices[1]] # Slice out an input sequence
            first_month = input_df.MONTH_TS.iloc[0]
            seq_name = f'{auto_id}_{first_month}'
            
            input_df = pd.DataFrame(self.scaler.transform(input_df), columns=input_df.columns)
            x = input_df[self.col_dict['input']] 
            x_obs = x.shift(1).fillna(x.iloc[0]).values
            x = x.values
            m = input_df[self.col_dict['missing']].values
            t = input_df[self.col_dict['delta']].values
            
            return seq_name, np.array([x[:,2:], m[:,1:], t[:,1:], x_obs[:,2:]]) # remove auto id and month_ts

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

  
    def write_to_disk(self, datadir, train_size):
        """
        IBDSequenceGenerator.write_to_disk:
         - Reads padded X, M, T matrices
         - Generate overlapping sequences and {tgt_type} training labels for each patient
         - Persist sequences as numpy arrays at data/{datadir}/{train, valid, test}/{AUTO_ID}_{START_TS}.npy
         - Persist labels for all sequences in data/{datadir}/{train, valid, test}/label_dict.pkl
         - Persist empirical mean values at data/{datadir}/x_mean.pkl
        """
        print(IBDSequenceGenerator.write_to_disk.__doc__)

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
                
        dataset, self.col_dict, x_mean = clean_longitudinal_inputs(load_pkl=True) # Load inputs
        print("Inputs loaded")
    
        groups = dataset.groupby('AUTO_ID')

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



class PatientDataset(data.Dataset):  
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

