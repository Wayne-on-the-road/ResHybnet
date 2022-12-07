# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:58:54 2022

@author: Wayne
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv


class ResHybNet(nn.Module):
    def __init__(self,input_dim, output_dim, cnn='' ,gnn='',residual='' ):
        super(ResHybNet,self).__init__()
        
        if cnn=='':
            self.mode= 'gnn'
        elif gnn=='':
            self.mode= 'cnn'
        else:
            self.mode= 'hybrid'
        #initialize cnn layer base on model type
        if cnn=='CNN':
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),                              
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)      
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),                               
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)      
            )
        # initialize gnn mode based on model type:
        if gnn== 'GCN':
            self.gnn1= GCNConv(input_dim, 16)
            self.gnn2= GCNConv(16, output_dim)
        elif gnn=='GAT':
            self.gnn1= GATConv(input_dim, 16)
            self.gnn2= GATConv(16, output_dim)
        elif gnn=='SAGE':
            self.gnn1= SAGEConv(input_dim, 16)
            self.gnn2= SAGEConv(16, output_dim)
            
        # initialize output fc layer based on feature number:      
        if input_dim == 5:
            self.output = nn.Linear(32,2)
            self.output_g= nn.Linear(5, 2)
        else:
            self.output = nn.Linear(64,2)
            self.output_g= nn.Linear(10, 2)
            
        if residual=='YES':
            self.res= True
        elif residual == 'NO':
            self.res= False
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.mode== 'cnn':
            x_c = torch.unsqueeze(x,dim=1)
            x_c = self.conv1(x_c)                  
            x_c = self.conv2(x_c)                
            x_c = x_c.view(x_c.size(0),-1)       
            out = self.output(x_c)
            
        elif self.mode== 'gnn':
            x_g = self.gnn1(x, edge_index)
            x_g = F.relu(x_g)
            x_g = F.dropout(x_g, training=self.training)
            x_g = self.gnn2(x_g, edge_index)
            out = self.output_g(x_g)

        else:
            x_g = self.gnn1(x, edge_index)
            x_g = F.relu(x_g)
            x_g = F.dropout(x_g, training=self.training)
            x_g = self.gnn2(x_g, edge_index)
            
            
            # add two channel
            if self.res:
                x_dual= x+x_g
            else:
                x_dual= x_g
            
            x_dual = torch.unsqueeze(x_dual,dim=1)
            x_dual = self.conv1(x_dual)                  
            x_dual = self.conv2(x_dual)                
            x_dual = x_dual.view(x_dual.size(0),-1)
            out= self.output(x_dual)

        return F.softmax(out,dim=1)



class c_CNN(nn.Module):
    def __init__(self,input_dim):
        super(c_CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                              
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)      
        )
        
        if input_dim == 5:
            self.output = nn.Linear(32,2)
        else:
            self.output = nn.Linear(64,2)
        

    def forward(self, x):
        out = self.conv1(x)                  
        out = self.conv2(out)                
        out = out.view(out.size(0),-1)       
        out = self.output(out)
        return F.softmax(out,dim=1)

  

  
