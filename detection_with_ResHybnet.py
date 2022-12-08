# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:58:54 2022

@author: Wayne
"""

import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
#import torch.utils.data as Data
import numpy as np
import pandas as pd
#from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,accuracy_score,roc_curve,accuracy_score,auc
import csv
from torch_geometric.data import Data
#import torchvision.models.resnet
from tool_and_model.early_stop_v1 import EarlyStopping
from tqdm import trange
from tool_and_model.hybrid_models import ResHybNet


LR= 0.01
train_round=10
dic_draw={}
EPOCH= 1000


# feature combination: 
f1_L= ['manual'] # manual feature

f2_L= ['lstm_seq74'] # automatic feature from different sequence length settings

#f2_L= ['lstm_seq5','lstm_seq16','lstm_seq27','lstm_seq38','lstm_seq49','lstm_seq60','lstm_seq74']

feat_list=[] # combine feature groups together
for f1 in f1_L:
    for f2 in f2_L:
        feat_list.append(f1 + '+' + f2)

#feat_list=['manual','lstm_seq74']

#component for hybnet:       
CNN='CNN'  # 'CNN' or ''
GNN='GCN'  # 'GCN','GAT','SAGE' or ''
Residual= 'YES' # 'YES' or 'NO'

#dir for all sample data and other needed files:
file_dir= './sample_data'
all_result= './detection_result'
# dir to store result for this setting:
result_dir= os.path.join(all_result,'ResHybnet_compare_sequence_length_10round') 
os.makedirs(result_dir,exist_ok=True)

# load edge information for the sample graph:   
df_e= pd.read_csv(os.path.join(file_dir,'1-data-test-undirected_edge.csv'))
edge= df_e.to_numpy().T
edge_index = torch.from_numpy(edge)                         


for feat in  feat_list:
    
    # csv file to store the final performance result for each model and feature combination:
    perform_file=os.path.join(result_dir,f'{CNN}_{GNN}_{feat}_{Residual}_result.csv')
    with open(perform_file, 'w', newline='') as f:
        writer = csv.writer(f)
        my_list = ['model', 'Fearures', 'round','test_acc','test_pre','test_rec','test_f1','epoch']
        writer.writerow(my_list)
    plot_file=perform_file.replace('results.csv','plot.npy')
    
    # load feature combination data:    
    df_data= pd.read_csv(os.path.join(file_dir,'1-data-test-'+ feat + '.csv'))
    
    # Input for x, [node_number,feature_dim]
    cols= df_data.columns[5:]
    
    x = df_data[cols].to_numpy(dtype=np.float32)
    x = torch.from_numpy(x)  # 
    
    # prepare lable y
    y = df_data['label'].to_numpy()
    y = torch.from_numpy(y)                               
        
    # pack x,y into pyg special data class
    data = Data(x=x,
                edge_index=edge_index,
                y=y)
    
    # seperate dataset 
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.train_mask[:data.num_nodes - 572] = 1                  
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.test_mask[data.num_nodes - 572:] = 1                    # 
    #data.num_classes = 10 # output dimension for GNN component, need to change it according to input feature dimonsion 
    
    for r in trange(train_round):
        
        # save train and test acc, loss, for the purpose of plotting after each round:
        train_loss_s = []
        train_acc_s = []
        train_rec_s = []
        train_pre_s = []
        train_f1_s = []
    
        test_loss_s = []
        test_acc_s = []
        test_rec_s = []
        test_pre_s = []
        test_f1_s = []
    
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResHybNet(input_dim=data.num_node_features, output_dim=data.num_node_features,cnn=CNN,gnn=GNN,residual=Residual).to(device)         # Initialize model
        data = data.to(device)                                                       
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # Initialize optimizer and training params
    
        # initiate early_stop parameter:
        best_model_path = os.path.join(result_dir,"early_stop_model" )
        os.makedirs(best_model_path,exist_ok=True)
        best_model_path = os.path.join(best_model_path,f'{CNN}_{GNN}_{feat}_{Residual}_{r}round_best.pt')
        early_stopping = EarlyStopping(save_path=best_model_path,verbose=(True),patience=30,delta=0.0001,metric='loss')
        #start training for this round:
        for epoch in range(EPOCH):
            model.train()
            optimizer.zero_grad()
            # Get output
            out = model(data)
            _, out1 = out.max(dim=1)
            pred_y = torch.masked_select(out1, data.train_mask.bool()).tolist()
            true_y = data.y[data.train_mask.bool()].tolist()
            train_acc_s.append(accuracy_score(true_y, pred_y))
        # wayne's debug:
            #print(out)    
            # Get loss
            loss = F.nll_loss(out[data.train_mask.bool()], data.y[data.train_mask.bool()].long())
            #print('loss',loss.item())
            # Backward
            loss.backward()
            optimizer.step()
            train_loss_s.append(loss.item())
           
            # after each epoch, evaluate it on test set:
            model.eval()
            out = model(data)
            _, out1 = out.max(dim=1)
            pred_y = torch.masked_select(out1, data.test_mask.bool()).tolist()
            true_y = data.y[data.test_mask.bool()].tolist()
            test_acc = accuracy_score(true_y, pred_y)
            test_acc_s.append(test_acc)
            # Get test loss
            test_loss = F.nll_loss(out[data.test_mask.bool()], data.y[data.test_mask.bool()].long())
            test_loss_s.append(test_loss.item())
            #print('epoch:',epoch,'test_loss:',test_loss.item(),'test_acc:',test_acc)
            
            # verify whether to do early stop:
            early_stopping(test_loss,model)
            if early_stopping.early_stop:
                print("Early stopping at epoch:",epoch)
                #print(accuracy_score(true_y, pred_y),precision_score(true_y, pred_y),recall_score(true_y, pred_y),f1_score(true_y, pred_y))
                break
        
        early_stopping.draw_trend(train_loss_s, test_loss_s)
        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        out = model(data)
        _, out1 = out.max(dim=1)
        pred_y = torch.masked_select(out1, data.test_mask.bool()).tolist()
        true_y = data.y[data.test_mask.bool()].tolist()
        test_acc = accuracy_score(true_y, pred_y)
        test_pre = precision_score(true_y, pred_y)
        test_rec = recall_score(true_y, pred_y)
        test_f1 = f1_score(true_y, pred_y)
        print(f'best model testing performance for round {r}:',test_acc,test_pre,test_rec,test_f1)
        # save the best result for this round:
        
        feature= feat
    
        with open(perform_file, 'a', newline='') as f:
            writer = csv.writer(f)
            my_list = [f'{CNN}_{GNN}',feature, r , test_acc, test_pre, test_rec, test_f1,epoch]
            writer.writerow(my_list)
        print('finished round:',r)
    
