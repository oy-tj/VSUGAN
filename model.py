import sys
sys.path.append('/opt/anaconda3/lib/python38.zip')
sys.path.append('/opt/anaconda3/lib/python3.8')
sys.path.append('/opt/anaconda3/lib/python3.8/lib-dynload')
sys.path.append('/home/oytj/.local/lib/python3.8/site-packages')
sys.path.append('/opt/anaconda3/lib/python3.8/site-packages')

import librosa
import librosa.display
from matplotlib import pyplot as plt 
import IPython
import math
import numpy as np
import os
import random
import time
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import pyworld
import pysptk


N_FFT = 512
def wav2spectrum(wav):
    S = librosa.stft(wav, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S
def spectrum2wav(spectrum):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    return x

class Generator(nn.Module):
    """G"""
    def __init__(self):
        super().__init__()
        # size notations = [feature_maps x width] (height omitted - 1D convolutions)
        # encoder gets a noisy signal and a clean signal as input   
        #noize cnn in[257,513]
        self.encN1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)   # out : [B x 16 x 129 x 257]
        self.encN1_nl = nn.PReLU()  # non-linear transformation after encoder layer 1
        self.encN2 = nn.Conv2d(16, 32, 3, 2, 1)  # [B x 32 x 65 x 129]
        self.encN2_nl = nn.ReLU()
        self.encN3 = nn.Conv2d(32, 64, 3, 2, 1)  # [B x 64 x 33 x 65]
        self.encN3_nl = nn.ReLU()
        self.encN4 = nn.Conv2d(64, 128, 3, 2, 1)  # [B x 128 x 17 x 33]
        self.encN4_nl = nn.ReLU()
        self.encN5 = nn.Conv2d(128,256, 3, 2, 1)  # [B x 256 x 9 x 17]
        self.encN5_nl = nn.ReLU()
        self.encN6 = nn.Conv2d(256,512, 3, 2, 1)  # [B x 512 x 5 x 9]
        self.encN6_nl = nn.ReLU()
        self.encN7 = nn.Conv2d(512,1024, 3, 2, 1)  # [B x 1024 x 3 x 5]
        self.encN7_nl = nn.ReLU()
        self.encN8 = nn.Conv2d(1024,2048, 3, 2, 1)  # [B x 2048 x 2 x 3]
        self.encN8_nl = nn.ReLU()
        
        #clean_voc cnn in[257,513]
        self.encC1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)   # out : [B x 16 x 129 x 257]
        self.encC1_nl = nn.ReLU()  # non-linear transformation after encoder layer 1
        self.encC2 = nn.Conv2d(16, 32, 3, 2, 1)  # [B x 32 x 65 x 129]
        self.encC2_nl = nn.ReLU()
        self.encC3 = nn.Conv2d(32, 64, 3, 2, 1)  # [B x 64 x 33 x 65]
        self.encC3_nl = nn.ReLU()
        self.encC4 = nn.Conv2d(64, 128, 3, 2, 1)  # [B x 128 x 17 x 33]
        self.encC4_nl = nn.ReLU()
        self.encC5 = nn.Conv2d(128,256, 3, 2, 1)  # [B x 256 x 9 x 17]
        self.encC5_nl = nn.ReLU()
        self.encC6 = nn.Conv2d(256,512, 3, 2, 1)  # [B x 512 x 5 x 9]
        self.encC6_nl = nn.ReLU()
        self.encC7 = nn.Conv2d(512,1024, 3, 2, 1)  # [B x 1024 x 3 x 5]
        self.encC7_nl = nn.ReLU()
        self.encC8 = nn.Conv2d(1024,2048, 3, 2, 1)  # [B x 2048 x 2 x 3]
        self.encC8_nl = nn.ReLU()
        
        

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homolgous encoder output,
        # so the feature map sizes are doubled
        #[B x 2048 x 2 x 3]
        self.dec9 = nn.ConvTranspose2d(in_channels=4096, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.dec9_nl = nn.ReLU()  # out : [B x 4096 x 2 x 3] -> (concat) [B x 1024 x 3 x 5]
        self.dec8 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.dec8_nl = nn.ReLU()  # out : [B x 2048 x 3 x 5] -> (concat) [B x 512 x 5 x 9]
        self.dec7 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.dec7_nl = nn.ReLU()  # out : [B x 1024 x 5 x 9] -> (concat) [B x 256 x 9 x 17]
        self.dec6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.dec6_nl = nn.ReLU()  # out : [B x 512 x 9 x 17] -> (concat) [B x 128 x 17 x 33]
        self.dec5 = nn.ConvTranspose2d(256, 64, 3, 2, 1)  # [B x 64 x 33 x 65]
        self.dec5_nl = nn.ReLU()
        self.dec4 = nn.ConvTranspose2d(128, 32, 3, 2, 1)  # [B x 32 x 65 x 129]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose2d(64, 16, 3, 2, 1)  # [B x 16 x 129 x 257]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose2d(32, 8, 3, 2, 1)  # [B x 8 x 257 x 513]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose2d(8, 1, 3, 1, 1)  # [B x 1 x 257 x 513]
        self.dec_tanh = nn.Tanh()





    def forward(self, nx, cx):
        """
        Forward pass of generator.

        Args:
            nx: input batch (signal+noise)
            cx: input batch (other clean signal)
        """
        ### encoding step
        Ne1 = self.encN1(nx)
        Ne2 = self.encN2(self.encN1_nl(Ne1))
        Ne3 = self.encN3(self.encN2_nl(Ne2))
        Ne4 = self.encN4(self.encN3_nl(Ne3))
        Ne5 = self.encN5(self.encN4_nl(Ne4))
        Ne6 = self.encN6(self.encN5_nl(Ne5))
        Ne7 = self.encN7(self.encN6_nl(Ne6))
        Ne8 = self.encN8(self.encN7_nl(Ne7))
        # c = compressed feature, the 'thought vector'
        Nc = self.encN8_nl(Ne8)
        ### encoding step
        Ce1 = self.encC1(cx)
        Ce2 = self.encC2(self.encC1_nl(Ce1))
        Ce3 = self.encC3(self.encC2_nl(Ce2))
        Ce4 = self.encC4(self.encC3_nl(Ce3))
        Ce5 = self.encC5(self.encC4_nl(Ce4))
        Ce6 = self.encC6(self.encC5_nl(Ce5))
        Ce7 = self.encC7(self.encC6_nl(Ce6))
        Ce8 = self.encC8(self.encC7_nl(Ce7))
        # c = compressed feature, the 'thought vector'
        Cc = self.encC8_nl(Ce8)
    
        # concatenate the thought vector with latent variable
        encoded = torch.cat((Nc, Cc), dim=1)

        ### decoding step
        d9 = self.dec9(encoded)
        # dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d9_c = self.dec8_nl(torch.cat((d9, Ne7), dim=1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, Ne6), dim=1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, Ne5), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, Ne4), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, Ne3), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, Ne2), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, Ne1), dim=1))
        d2 = self.dec2(d3_c)
        d1 = self.dec1(d2)
        out = self.dec_tanh(d1)
        return out
    
class Discriminator(nn.Module):
    """D"""
    def __init__(self, dropout_drop=0.5):
        super().__init__()
        # Define convolution operations.
        # (#input channel, #output channel, kernel_size, stride, padding)
        #input 維度 [B, 2, 257, 513]
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),  # [32, 129, 257]
            nn.BatchNorm2d(32),#归一化
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1), # [64, 65, 129]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1), # [128, 33, 65]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 2, 1), # [256, 17, 33]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, 2, 1), # [512, 9, 17]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 1024, 3, 2, 1), # [1024, 5, 9]
            nn.BatchNorm2d(1024),
            nn.ReLU(),

        )
        self.dnn = nn.Sequential(
            nn.Linear(1024*5*9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, gen_output,xn):
        x=torch.cat((gen_output, xn), dim=1)
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)#把tenor拉直放入DNN
        out = self.dnn(out)
        return self.sigmoid(out)