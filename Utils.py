import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

import Models
from Models import *

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