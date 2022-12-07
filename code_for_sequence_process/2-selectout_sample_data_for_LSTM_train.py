# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:22:50 2022

@author: Wayne
"""

import pandas as pd
import os
from tqdm import tqdm,trange

filepath=r'.\cert4.2_data'
sample_path= r'.\sample_data'
    
user_file_dir=os.path.join(filepath,'user_file')

day_seq_dir= os.path.join(user_file_dir,'day_seq_dir')

sample_data_path= os.path.join(sample_path,'1-data-test-combine.csv')

# load the user-day sample dataset used in previous study, 1908 node,and extract head for further LSTM train:
df_data = pd.read_csv(sample_data_path)

df_head = df_data[['date_index','user_index','date','label']]


# select out the user-day used in previous sample dataset for sequential activity feature extraction:

list_new= []   

for i in trange(df_head.shape[0]):
    df_user = pd.read_csv(os.path.join(day_seq_dir,df_head.iloc[i]['user_index']+'.csv'))
    head_list= df_head.iloc[i,:].to_list()  
    df_slice = df_user.iloc[head_list[0],2:].to_list()
    head_list.extend(df_slice)
    list_new.append(head_list)

df_new=pd.DataFrame(list_new)
df_new.rename(columns={0:'date_index',1:'user_index',2:'date',3:'label'},inplace=True)
df_new.to_csv(sample_data_path.replace('.csv','-sequence.csv'),index=False)
    

# for row in df_head.iterrows():
#     df_user = pd.read_csv(os.path.join(day_seq_dir,row['user_index']))
#     user_tuple= df_head.itertuples()
    