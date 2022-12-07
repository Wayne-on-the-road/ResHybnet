# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:15:31 2022

@author: Wayne
"""
import pandas as pd
import os
import datetime 
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm,trange
import csv

def convert_time(path):
    df= pd.read_csv(path)
    df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y %H:%M:%S')
    return df
        
def create_user_files(filepath,file_name,user_file_path,u_list):
    #L_df= []
    df_dic= {}
    for file in file_name:
        df= pd.read_csv(os.path.join(filepath, file+'.csv'))
        df_dic.update({file:df})    
    
    for user in tqdm(u_list):
        user_path= os.path.join(user_file_path, user+'.csv')
        df_user= pd.read_csv(user_path)
        for key in df_dic.keys():
            df= df_dic[key]
            df= df[df['user'] == user]
            df['file']= key
            df_user= pd.concat([df_user,df],axis=0)
        df_user.sort_values(by='date',inplace= True, ascending=True)
        df_user.to_csv(user_path,index=False)

def encode_row(row):
    act_list=[]
    date= pd.to_datetime(row.date,format='%m/%d/%Y %H:%M:%S')
    #h_delta= datetime.timedelta(hours=1)
    act_list.append('work' if (date.hour >= 8) and (date.hour < 18) else 'off')
    act_list.append('' if pd.isnull(row.activity) else row.activity)
    act_list.append('' if pd.isnull(row.filename) else row.filename[-3:])# return last three string to indicate file type
    
    if not pd.isnull(row.to):
       act_list.append('intern' if row.to.find('@dtaa') > 0 else 'extern') 
    
    act_str= ''.join(act_list)
    dic= {'workLogon':1,'workLogoff':2,'workConnect':3,'workDisconnect':4,'workdoc':5,'workexe':6,
          'workjpg':7,'workpdf':8,'worktxt':9,'workzip':10,'workintern':11,'workextern':12,
          
          'offLogon':13,'offLogoff':14,'offConnect':15,'offDisconnect':16,'offdoc':17,'offexe':18,
                'offjpg':19,'offpdf':20,'offtxt':21,'offzip':22,'offintern':23,'offextern':24}    
    
    return dic[act_str]


def encode_user(user_file_dir,u_list):#take in user list and the dir that store user behavior csv file, give each action a code
    for user in tqdm(u_list):
        code_list=[]
        user_path= os.path.join(user_file_dir, user+'.csv')
        df_user= pd.read_csv(user_path)
        for row in df_user.itertuples():
            row_code= encode_row(row)
            code_list.append(row_code)
        df_user['encode']=code_list
        df_user.to_csv(user_path,index= False)
    return
        
def to_day_sequence(user_file_dir,u_list,d_list):
    
    d_delta= datetime.timedelta(days=1)
    new_user_dir = os.path.join(user_file_dir,'day_seq_dir')
    os.makedirs(new_user_dir,exist_ok=True)
    for user in tqdm(u_list):
        L=[]
        user_path= os.path.join(user_file_dir, user+'.csv')
        df_user= convert_time(user_path)
        for day in d_list:
            df= df_user[(day <= df_user['date']) & (df_user['date']< day + d_delta)]
            row= df['encode'].to_list()
            row.insert(0,user)
            row.insert(0,day)
            L.append(row)          
        df_new=pd.DataFrame(L)
        df_new.rename(columns={0:'date',1:'user'},inplace=True)
        df_new.to_csv(os.path.join(new_user_dir, user+'.csv'),index=False)
        #seq_len.append(df_new.shape[1])
     
    return
        
def count_seq_lenth(user_file_dir,u_list):
    L=[]
    for user in u_list:
        user_path= os.path.join(user_file_dir, user+'.csv')
        df_user= pd.read_csv(user_path)
        L.append(df_user.shape[1])
    
    return L    
        
if __name__ == '__main__':
    #mp.freeze_support()
    
    
    # #generate date_list for extracting feature:only weekday
    date= pd.date_range(start= '2010-01-04', end= '2011-05-16')
    date= date.tolist()
    date= pd.DataFrame({'date':date})
    date['date']= pd.to_datetime(date['date'],format='%Y-%m-%d %H:%M:%S')
    date['week']= date['date'].dt.dayofweek
    date= date[date['week'].isin([0,1,2,3,4])] 
    date= date.drop(labels='week',axis=1)
    date_list= date['date'].to_list()
    #

    filepath=r'.\cert4.2_data'
    
    user_file_dir=os.path.join(filepath,'user_file')

    

    #generate user list for extracting feature:multiprocessing based
    
    u= pd.read_csv(os.path.join(filepath,'userlist.csv'))
    u_list= u['user_id'].to_list()
    
    #create empty csv file with head for each user:
    head= ['file','id','date','user','pc']
    os.makedirs(user_file_dir,exist_ok=True)
    for u in u_list:
        df= pd.DataFrame(columns=head)
        df.to_csv(os.path.join(user_file_dir,u+'.csv'),index= False)
        
    # pick the activity log files to extract activities from, here we used four files:
    file_name=['logon','device','file','email']

    #create_user_files(filepath,file_name, user_file_dir, u_list)
    #encode_user(user_file_dir,u_list)
    #to_day_sequence(user_file_dir, u_list, date_list)
    
    #seperate user into group for multiprocessing:
    num_u= len(u_list)
    num_p= int(mp.cpu_count()/2)
    block= int(num_u/num_p)
    #block= 1
    L_block= []
    for i in range(0,num_u,block):
        block_users= u_list[i:i+block]
        L_block.append(block_users)
        
        
    # step1:fill each user file with activities extracted from raw data: 
    pool= mp.Pool(num_p)
    start0= time.time()
    for block_users in L_block:
        pool.apply_async(create_user_files,args=(filepath,file_name, user_file_dir, block_users))
    pool.close()
    pool.join()
    end0= time.time()
    print('step1 time used:',end0-start0)
    
    # step2:encode all the activities in each user file:
    pool= mp.Pool(num_p)
    start0= time.time()
    for block_users in L_block:
        pool.apply_async(encode_user,args=(user_file_dir, block_users))
    pool.close()
    pool.join()
    end0= time.time()
    print('step2 time used:',end0-start0)
    
    # step3: reconstruct all user activities file into daily sequence based:
    pool= mp.Pool(num_p)
    start0= time.time()
    for block_users in L_block:
        pool.apply_async(to_day_sequence,args=(user_file_dir, block_users,date_list))
    pool.close()
    pool.join()
    end0= time.time()
    print('step3 time used:',end0-start0)

    
    # # count sequence_length data:
    # seq_len= count_seq_lenth(os.path.join(user_file_dir,'day_seq_dir'), u_list)
    # seq_stat=pd.DataFrame(seq_len)
    # seq_stat.to_csv(os.path.join(user_file_dir,'day_seq_dir','seq_stat.csv'))
    

    
