# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:25:04 2022

@author: Wayne
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path= './default_best.pt', patience=7, verbose=False, delta=0.000001,metric='loss'):
        """
        Args:
            save_path : path for model to be saved
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        #self.save_name = save_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.metric= metric
        self.val_bound = np.Inf
        self.sign = -1 if metric == 'loss' else 1
        #os.makedirs(save_path,exist_ok=True)

    def __call__(self, value, model):
                   
            score = value*self.sign
    
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(value, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(value, model)
                self.counter = 0
                


    def save_checkpoint(self, value, model):
        '''Saves model when metric decrease or increase '''
        if self.verbose:
            if self.sign == -1:
                print(f'Validation {self.metric} Decreased ({self.val_bound:.6f} --> {value:.6f}).  Saving model ...')
            else:
                print(f'Validation {self.metric} Increased ({self.val_bound:.6f} --> {value:.6f}).  Saving model ...')
        #path = os.path.join(self.save_path)
        torch.save(model.state_dict(), self.save_path)	
        self.val_bound = value

    def draw_trend(self,train_list,test_list):
        """help to draw the value(loss,acc and so on) trend and the stopping point"""
        
        plt.plot(range(1,len(train_list)+1),train_list, label='Training'+ self.metric)
        plt.plot(range(1,len(test_list)+1),test_list,label='Validation' + self.metric)
        
        # find position of check point,-1 means this a minimize problem like loss or cost
        if self.sign == -1:
            checkpoint = test_list.index(min(test_list))+1 
        else:
            checkpoint = test_list.index(max(test_list))+1
        
        plt.axvline(checkpoint, linestyle='--', color='r',label='Early Stopping Checkpoint')
        
        plt.xlabel('epochs')
        plt.ylabel(self.metric)
        plt.ylim(min(test_list),max(test_list)) # consistent scale
        plt.xlim(0, len(test_list)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
            

        
