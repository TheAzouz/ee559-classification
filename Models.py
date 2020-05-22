import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

from Utils import data_loading_standerdize

class BaselineNet(nn.Module):
    def __init__(self, nb_hidden=100):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

class FullyConnectedNet(nn.Module):
    def __init__(self, nb_hidden=100):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(392, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)
        self.act = nn.Tanh()


    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1,392)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x= self.act(x)
        return x


######################################################################

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size,skip_connections,batch_normalization):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)
        self.skip_connections = skip_connections
        self.batch_normalization= batch_normalization

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections : y = y + x
        y = F.relu(y)

        return y

class ResNet(nn.Module):
    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size = 3, nb_classes = 10,skip_connections=True,batch_normalization=True):
        super(ResNet, self).__init__()

        self.conv = nn.Conv2d(2, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)
        self.skip_connections = skip_connections
        self.batch_normalization= batch_normalization        

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size,skip_connections,batch_normalization)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(nb_channels, nb_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.resnet_blocks(x)
        x = F.avg_pool2d(x, 14).view(x.size(0), -1)
        x = self.fc(x)
        return x

######################################################################

class LeNet(nn.Module):
    def __init__(self, weight_sharing = False, activation_conv = 'tanh', activation_fully = 'tanh'):
        super(LeNet, self).__init__()
        
        #Image1_input
        self.conv1_0 = nn.Conv2d(1, 6, kernel_size=5)   # (14 => 12) + average pool => 6
        self.conv2_0 = nn.Conv2d(6, 16, kernel_size=5)  # (6 => 4) + average pool => 2
        self.conv3_0 = nn.Conv2d(16, 120, kernel_size=5) # (2 => 1)
        self.fc1_0 = nn.Linear(120, 84)
        self.fc2_0 = nn.Linear(84, 10)
        
        # Image2_input
        
        self.conv1_1 = nn.Conv2d(1, 6, kernel_size=5) 
        self.conv2_1 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3_1 = nn.Conv2d(16, 120, kernel_size=5) 
        self.fc1_1 = nn.Linear(120, 84)
        self.fc2_1 = nn.Linear(84, 10)
        
        # Comparison
        self.fc3 = nn.Linear(20, 2)
        self.weight_sharing = weight_sharing
        
        if activation_conv == 'Relu':
            self.act_conv = F.relu
        elif activation_conv == 'LeakyRelu':
            self.act_conv = F.leaky_relu
        elif activation_conv == 'Sigmoid':
            self.act_conv = F.sigmoid
        else:
            self.act_conv = nn.Tanh()
            
        
        if activation_fully == 'Relu':
            self.act_fully = F.relu
        elif activation_fully == 'LeakyRelu':
            self.act_fully = F.leaky_relu
        elif activation_fully == 'Sigmoid':
            self.act_fully = nn.Sigmoid()
        else:
            self.act_fully = nn.Tanh()
            
        self.upsample = nn.Upsample(size = (32,32), align_corners=True, mode= 'bilinear')
            
    def forward(self, x ):
        
        input_1 = self.upsample(x[:,0,:].view(-1,1,14,14)) # Selecting first image
        input_2 = self.upsample(x[:,1,:].view(-1,1,14,14)) # Selectind second image
        
        
        input_1 = self.act_conv(self.conv1_0(input_1)) # 1st Convolution Layer
        input_1 = self.act_conv(F.avg_pool2d(input_1, kernel_size=2, stride=2)) # 1st Averaging pool
        input_1 = self.act_conv(self.conv2_0(input_1)) # 2nd Convolution Layer
        input_1 = self.act_conv(F.avg_pool2d(input_1, kernel_size=2, stride=2)) # 2nd Averaging pool
        input_1 = self.act_conv(self.conv3_0(input_1)) # 3nd Convolution Layer
        
        input_1 = self.act_fully(self.fc1_0(input_1.view(-1,120)))
        input_1 = self.fc2_0(input_1)
        
        
        if self.weight_sharing :
            
            input_2 = self.act_conv(self.conv1_0(input_2)) # 1st Convolution Layer
            input_2 = self.act_conv(F.avg_pool2d(input_2, kernel_size=2, stride=2)) # 1st Averaging pool
            input_2 = self.act_conv(self.conv2_0(input_2)) # 2nd Convolution Layer
            input_2 = self.act_conv(F.avg_pool2d(input_2, kernel_size=2, stride=2)) # 2nd Averaging pool
            input_2 = self.act_conv(self.conv3_0(input_2)) # 3nd Convolution Layer

            input_2 = self.act_fully(self.fc1_0(input_2.view(-1,120)))
            input_2 = self.fc2_0(input_2)
            
        else:
            
            input_2 = self.act_conv(self.conv1_1(input_2)) # 1st Convolution Layer
            input_2 = self.act_conv(F.avg_pool2d(input_2, kernel_size=2, stride=2)) # 1st Averaging  pool
            input_2 = self.act_conv(self.conv2_1(input_2)) # 2nd Convolution Layer
            input_2 = self.act_conv(F.avg_pool2d(input_2, kernel_size=2, stride=2)) # 2nd Averaging  pool
            input_2 = self.act_conv(self.conv3_1(input_2)) # 3nd Convolution Layer

            input_2 = self.act_fully(self.fc1_1(input_2.view(-1,120)))
            input_2 = self.fc2_1(input_2)
        
        
        #Comparison layer
        concat = torch.cat((input_1,input_2),1)
        x = self.fc3(concat)
        
        return input_1,input_2,x


class CNNet(nn.Module):
    def __init__(self, weight_sharing = False, activation_conv = 'tanh', activation_fully = 'tanh'):
        super(CNNet, self).__init__()
        
        #Image1_input
        self.conv1_0 = nn.Conv2d(1, 32, kernel_size=3)   # (14 => 12) + avgpool => 6
        self.conv2_0 = nn.Conv2d(32, 64, kernel_size=3)  # (6 => 4) + avgpool => 2
        self.fc1_0 = nn.Linear(256, 120) 
        self.fc2_0 = nn.Linear(120, 84)
        self.fc3_0 = nn.Linear(84, 10)
        
        # Image2_input
        
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(256, 120) 
        self.fc2_1 = nn.Linear(120, 84)
        self.fc3_1 = nn.Linear(84, 10)
        
        # Comparison
        self.fc3 = nn.Linear(20, 2)
        self.weight_sharing = weight_sharing
        
        if activation_conv == 'Relu':
            self.act_conv = F.relu
        elif activation_conv == 'LeakyRelu':
            self.act_conv = F.leaky_relu
        elif activation_conv == 'Sigmoid':
            self.act_conv = F.sigmoid
        else:
            self.act_conv = nn.Tanh()
            
        
        if activation_fully == 'Relu':
            self.act_fully = F.relu
        elif activation_fully == 'LeakyRelu':
            self.act_fully = F.leaky_relu
        elif activation_fully == 'Sigmoid':
            self.act_fully = F.sigmoid
        else:
            self.act_fully = nn.Tanh()
            
    def forward(self, x ):
        
        input_1 = x[:,0,:].view(-1,1,14,14) # Selecting first image
        input_2 = x[:,1,:].view(-1,1,14,14) # Selectind second image
        
        
        input_1 = self.act_conv(self.conv1_0(input_1)) # 1st Convolution Layer
        input_1 = self.act_conv(F.max_pool2d(input_1, kernel_size=2, stride=2)) # 1st max pool
        input_1 = self.act_conv(self.conv2_0(input_1)) # 2nd Convolution Layer
        input_1 = self.act_conv(F.max_pool2d(input_1, kernel_size=2, stride=2)) # 2nd max pool
    
        input_1 = self.act_fully(self.fc1_0(input_1.view(-1,256)))
        input_1 = self.act_fully(self.fc2_0(input_1))
        input_1 = self.fc3_0(input_1)
        
        
        if self.weight_sharing :
            
            input_2 = self.act_conv(self.conv1_0(input_2)) # 1st Convolution Layer
            input_2 = self.act_conv(F.max_pool2d(input_2, kernel_size=2, stride=2)) # 1st max pool
            input_2 = self.act_conv(self.conv2_0(input_2)) # 2nd Convolution Layer
            input_2 = self.act_conv(F.max_pool2d(input_2, kernel_size=2, stride=2)) # 2nd max pool

            input_2 = self.act_fully(self.fc1_0(input_2.view(-1,256)))
            input_2 = self.act_fully(self.fc2_0(input_2))
            input_2 = self.fc3_0(input_2)
            
        else:
            
            input_2 = self.act_conv(self.conv1_1(input_2)) # 1st Convolution Layer
            input_2 = self.act_conv(F.max_pool2d(input_2, kernel_size=2, stride=2)) # 1st max pool
            input_2 = self.act_conv(self.conv2_1(input_2)) # 2nd Convolution Layer
            input_2 = self.act_conv(F.max_pool2d(input_2, kernel_size=2, stride=2)) # 2nd max pool

            input_2 = self.act_fully(self.fc1_1(input_2.view(-1,256)))
            input_2 = self.act_fully(self.fc2_1(input_2))
            input_2 = self.fc3_0(input_2)
        
        
        #Comparison layer
        concat = torch.cat((input_1,input_2),1)
        x = self.fc3(concat)
        
        return input_1,input_2,x
    
def compute_errors(Net,model,data,n_samples):
    nb_errors = 0
    if (Net == 'BaselineNet'or
                Net == 'FullyConnectedNet'or 
                Net == 'ResNet'):
        for input_,targets, _ in data:
            output = model(input_)
            _,predicted_classes = torch.max(output,1)
            for i in range(0,output.size(0)):
                if(predicted_classes[i]!=targets[i]):
                    nb_errors = nb_errors+1
    else:
        for input_,targets, _ in data:
            output = model(input_)
            _,predicted_classes = torch.max(output[2],1)
            for i in range(0,output[2].size(0)):
                if(predicted_classes[i]!=targets[i]):
                    nb_errors = nb_errors+1
                
    return (nb_errors/n_samples) * 100


def train_Net_model(Net = 'LeNet',
                    weight_sharing = False, auxilary_loss = False,
                    batch_size = 100, n_epochs = 25, lr = 0.2, 
                    verbose = False):
    
    if Net == 'CNN':
        model = CNNet(weight_sharing, activation_conv = 'tanh', activation_fully = 'tanh')
    elif Net == 'BaselineNet':
        model = BaselineNet()
    elif Net == 'FullyConnectedNet':
        model = FullyConnectedNet()
    elif Net == 'ResNet':
        model = ResNet(nb_residual_blocks = 15, nb_channels = 10, kernel_size = 3, nb_classes = 2)
    else: 
        model = LeNet(weight_sharing, activation_conv = 'tanh', activation_fully = 'tanh')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = lr)
    error = []
    train_error = []
    test_error = []
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = data_loading_standerdize(1000)

    for e in range(0,n_epochs):
        
        train_data = zip(train_input.split(batch_size),
                         train_target.split(batch_size),
                         train_classes.split(batch_size))
        
        test_data = zip(test_input.split(batch_size),
                        test_target.split(batch_size),
                        test_classes.split(batch_size))
        
        for inputs, targets, classes in train_data:
            output = model(inputs)
            
            if (Net == 'BaselineNet'or
                Net == 'FullyConnectedNet'or 
                Net == 'ResNet'):
                loss = criterion(output,targets)
            else:  
                loss_1 = criterion(output[0],classes[:,0])
                loss_2 = criterion(output[1],classes[:,1])
                main_loss = criterion(output[2],targets)

                # Select loss 
                if auxilary_loss:
                    loss = main_loss + 0.8*loss_1 + 0.8*loss_2
                else:
                    loss = main_loss
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        error.append(loss.item())
        
        train_data = zip(train_input.split(batch_size),
                         train_target.split(batch_size),
                         train_classes.split(batch_size))   
        
        train_error.append(compute_errors(Net,model,train_data,train_input.size(0)))
        test_error.append(compute_errors(Net,model,test_data,test_input.size(0)))
        
        if(e%5 ==0) and verbose:
            print('epoch : ',e,' loss : ',loss.item())
    
    print("Training accuracy = {0:.2f}%, Testing accuracy = {1:.2f}%\n".format(train_error[-1],test_error[-1]), )
               
    return model, error, train_error, test_error
