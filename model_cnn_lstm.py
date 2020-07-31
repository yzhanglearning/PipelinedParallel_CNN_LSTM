## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
## https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


import numpy as np
import scipy.misc as misc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t 


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1,10, kernel_size=3)
        self.conv2 = nn.Conv3d(10,20, kernel_size=3)
        self.conv3 = nn.Conv3d(20,40, kernel_size=2)
#        self.conv4 = nn.Conv3d(40,80, kernel_size=3)   
        self.drop = nn.Dropout3d()
        
        self.fc1 = nn.Linear(40*6*6*6, 1280)   # 4x4x4x80
        self.fc2 = nn.Linear(1280, 512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=None, padding=0)
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)

    def forward(self,x):
        x = self.batchnorm(x)
        x = self.relu(self.pool(self.conv1(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv2(x)))
        x = self.drop(x)
        x = self.relu(self.pool(self.conv3(x)))
        x = self.drop(x)
        
        print('after the convolutional layers, the shape is:'.format(x.shape))   # ([200, 40, 6, 6, 6])
        print(x.shape)
#        X = self.relu(self.pool(self.conv4(x)))
#        x = self.drop(x)
        x = x.view(-1, 40*6*6*6)
        print('before fc-layers, the shape is:'.format(x.shape))  ##
        print(x.shape)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        print('after fc-layers, the shape is:')
        print(x.shape)
        return x




    
## Assumption of input data dimension is: [batch_size, C, H, W, Z, seq_len]

## Assumption the dimension for PyTorch model is: [batch_size, seq_len, C, H, W, Z]




class CombineRNN(nn.Module):
    def __init__(self, devices):
        super(CombineRNN, self).__init__()

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices

        self.cnn = CNN().to(self.devices[0])
        self.hid_size = 256
        self.rnn = nn.LSTM(
            input_size = 512,
            hidden_size = self.hid_size,
            num_layers = 16,
            batch_first = True).to(self.devices[-1])
        self.linear1 = nn.Linear(self.hid_size,64).to(self.devices[-1])
        self.relu = nn.ReLU().to(self.devices[-1])
        self.linear2 = nn.Linear(64,1).to(self.devices[-1])

    def forward(self, x):

        x = x.permute(0,5,1,2,3,4)
        batch_size, time_steps, C, H, W, Z = x.size()  # batch, channel, height, width, depth, time
        #batch_size, C, H, W, Z, time_steps = x.size()


        #print(batch_size, H, W, Z, time_steps)
        
        c_in = x.reshape(batch_size*time_steps, C, H, W, Z).to(self.devices[0])  # batch_size*time_steps becomes dummy_batch)
        c_out = self.cnn(c_in)
        
        #r_in = c_out.view(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
        r_in = c_out.reshape(batch_size, time_steps, -1).to(self.devices[-1])
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.relu(self.linear1(r_out[:,-1,:]))
        r_output = self.linear2(r_out2)

        return r_output




    

class PipelineCombineRNN(CombineRNN):
    def __init__(self, devices, split_size):
        super(PipelineCombineRNN, self).__init__(devices)
        self.split_size = split_size

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices
        
    def forward(self, x):

        x = x.permute(0,5,1,2,3,4)
        
        splits = iter(x.split(self.split_size, dim=0))

        s_next = next(splits)

        batch_size, time_steps, C, H, W, Z = s_next.size()  # batch, channel, height, width, depth, time
        #batch_size, C, H, W, Z, time_steps = s_next.size()
        

        ## is this reshape correct? transpose (batch_size, time_steps, C, H, W, Z)
        ## numpy version of b = np.transpose(a, [0,1,2,4,6,3,5])
        ## pytorch version...
        ## s_next = s_next.permute(0,5,1,2,3,4)
        s_next = s_next.reshape(batch_size*time_steps, C, H, W, Z).to(self.devices[0])  # batch_size*time_steps becomes dummy_batch

        s_prev = self.cnn(s_next).to(self.devices[-1])
        ret = []

        for s_next in splits:
            s_prev = s_prev.reshape(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
            s_prev, _ = self.rnn(s_prev)
            s_prev = self.linear2(self.relu(self.linear1(s_prev[:,-1,:])))
            ret.append(s_prev)

            s_next = s_next.reshape(batch_size*time_steps, C, H, W, Z).to(self.devices[0])  # batch_size*time_steps becomes dummy_batch
            s_prev = self.cnn(s_next).to(self.devices[-1])
            
        s_prev = s_prev.reshape(batch_size, time_steps, -1)  # transform back to [batch_size, time_steps, feature]
        s_prev, _ = self.rnn(s_prev)
        s_prev = self.linear2(self.relu(self.linear1(s_prev[:,-1,:])))
        ret.append(s_prev)

        return torch.cat(ret)
    

















class ModelParallelCombineRNN(CombineRNN):
    def __init__(self, devices):
        super(ModelParallelCombineRNN, self).__init__()

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices

        self.batchnorm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        
        self.seq1 = nn.Sequential(
            self.batchnorm,
            self.cnn).to(self.devices[0])

        self.seq2 = nn.Sequential(
            self.rnn).to(self.devices[0])

        self.seq3 = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2).to(self.devices[1])
        #self.linear1.to(self.devices[1])
        #self.linear1.to(self.devices[1])
        #self.relu.to(self.devices[1])

        
    def forward(self, x):
        
        batch_size, C, H, W, Z, time_steps = x.size()  # batch, channel, height, width, depth, time
#        print(batch_size, H, W, Z, time_steps)
        
        c_in = x.view(batch_size*time_steps, C, H, W, Z)  # batch_size*time_steps becomes dummy_batch
        print(c_in.shape)
        # c_in = c_in.unsqueeze(1)
        c_out = self.seq1(c_in)
        print(c_out.shape)
#        c_out = c_out.squeeze(1)
        r_in = c_out.view(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
        r_out, (h_n, h_c) = self.seq2(r_in)
        r_out = r_out.to(self.devices[1])
        #r_out2 = self.relu(self.linear1(r_out[:,-1,:]))
        #r_output = self.linear2(r_out2)
        r_output = self.seq3(r_out[:,-1,:])
                                   
        return r_output








        
