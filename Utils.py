import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

from Models import *

import matplotlib.pyplot as plt

def data_loading_standerdize(n_samples):
    #Data loading
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    #Standarization of the training set
    mu,std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    #Standarization of the testing set
    mu,std = test_input.mean(), test_input.std()
    test_input.sub_(mu).div_(std)
    return train_input, train_target, train_classes, test_input, test_target, test_classes

def train_model_and_display_accuracy(Net, verbose=False,n_epochs=25,batch_size=100,lr=0.1):
    
    weight_sharing = [False,True]
    auxiliary_loss = [False,True]
        
    if Net == 'CNN' or Net == 'LeNet' :
        
        fig, axes = plt.subplots(2,2, figsize=(18, 12))
        for i,weight in enumerate(weight_sharing):
            for j,aux in enumerate(auxiliary_loss):

                print("Weight Sharing = '{0}', Auxilary Loss = '{1}'".format(weight,aux))

                model,error,train_error,test_error = train_Net_model(Net = Net,weight_sharing = weight,auxilary_loss = aux,
                                                                     n_epochs = 25, verbose = False, batch_size=100,lr=0.2)
                axes[i,j].plot(train_error, label='Training error')
                axes[i,j].plot(test_error, label='Testing error')
                axes[i,j].set_xlabel("Epoch")
                axes[i,j].set_ylabel("Accuracy")
                axes[i,j].legend(loc='upper right')
                axes[i,j].set_title("Weight Sharing = '{0}', Auxilary Loss = '{1}'".format(weight,aux))
            
    else:
        fig, ax = plt.subplots(1,1,figsize=(18, 12))
        ax.plot(train_error, label='Training error')
        ax.plot(test_error, label='Testing error')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc='upper right')
        axe.set_title(" {} ".format(Net))
        