import torch
import numpy as np
from torch import nn
from utils import *

class DVAE(nn.Module):
    def __init__(self, input_size, hidden_size_emission, hidden_size_transition, output_dim):
        self.encoder = Encoder(input_size, hidden_size_transition)
        self.decoder = Decoder(input_size, hidden_size_emission)

    def forward(self, x):
        pass



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        # DKS only
        self.combiner_linear = nn.Linear(latent_dim, hidden_size)
        self.latent_z = []

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden_right = hidden[0:hidden.shape[0]//2, :, :]
        hidden_left = hidden[hidden.shape[0]//2:, :, :]

    def sample_gaussian(self, m, v):
        epsilon = torch.randn(size=m.shape)
        z = epsilon*torch.sqrt(v) + m
        return z




class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass