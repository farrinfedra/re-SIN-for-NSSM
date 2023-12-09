import torch
import numpy as np
from torch import nn
from utils import *
import torch.nn.functional as F

class DVAE(nn.Module):
    def __init__(self, 
                    input_dim=88, 
                    hidden_dim=400,
                    hidden_dim_em=100, 
                    hidden_dim_tr=200, 
                    latent_dim=100,
                    dropout=0.1,
                    combiner_type='dks',
                    rnn_type='rnn',
                    
                    ):
        super(DVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_em = hidden_dim_em
        self.hidden_dim_tr = hidden_dim_tr
        self.latent_dim = latent_dim
        self.output_dim = input_dim
        self.dropout = dropout
        self.combiner_type = combiner_type
        self.rnn_type = rnn_type
        
        self.encoder = Inference(self.input_dim, 
                                self.hidden_dim, 
                                self.latent_dim,
                                self.dropout,
                                self.combiner_type,
                                self.rnn_type)
        
        self.decoder = Generator(self.hidden_dim_em, 
                                self.latent_dim, 
                                self.hidden_dim_tr, 
                                self.output_dim)

    def forward(self, x):
        z, mus_inference, sigmas_inference = self.encoder(x)
        x_hat, mus_generator, sigmas_generator = self.decoder(z)
        return x_hat, mus_inference, sigmas_inference, mus_generator, sigmas_generator


class DKSCombiner(nn.Module): #DKS
    def __init__(self, latent_dim, hidden_size):
        super(DKSCombiner, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        

        self.combiner = torch.nn.Linear(self.latent_dim, 
                                        self.hidden_size)
        self.mu_linear = torch.nn.Linear(self.hidden_size, 
                                            self.latent_dim)
        self.sigma_linear = torch.nn.Sequential(
                                            torch.nn.Linear(self.hidden_size, 
                                                            self.latent_dim),
                                            torch.nn.Softplus())
        self.mus = []
        self.sigmas = []
        
    def sample(self, mu, var): #epsilon ~ N(0, 1) * sqrt(var) + mean
        return mu + torch.sqrt(var) * torch.randn_like(mu)
        

    def forward(self, h_right, h_left=None):
        #shape Z: (batch_size, seq_len, latent_dim)
        if h_right is not None:
            b, seq_len, _ = h_right.shape
            device = h_right.device
            
        else:
            b, seq_len, _ = h_left.shape
            device = h_left.device
        # Z = torch.zeros(h_right.shape[0], h_right.shape[1] + 1, self.latent_dim).to(h_right.device)
        # mus, sigmas = torch.zeros((b, seq_len, self.latent_dim)), \
                        # torch.ones((b, seq_len, self.latent_dim))
        
        mus = []
        sigmas = []
        z_init = torch.zeros( (b, 1, self.latent_dim) ).to(device) #TODO: changed to 0
        Z = [z_init]
        
        for t in range(1, seq_len + 1):
            z_prev = Z[-1].squeeze(1) #shape: (batch_size, latent_dim)
            
            h_combined = self.combiner(z_prev)
            
            if h_right is None:
                assert h_left is not None
                h_combined = .5 * (F.tanh(h_combined) + h_left[:, t - 1, :])
                
            elif h_left is None:
                assert h_right is not None
                h_combined = .5 * (F.tanh(h_combined) + h_right[:, t - 1, :])
            else:
                h_combined = .3 * (F.tanh(h_combined) + h_right[:, t - 1, :] + h_left[:, t - 1, :])
            
            mu = self.mu_linear(h_combined) #shape: (batch_size, latent_dim)
            var = self.sigma_linear(h_combined)
            z_t = self.sample(mu, var)

            mus.append(mu.unsqueeze(1))
            sigmas.append(var.unsqueeze(1))
            Z.append(z_t.unsqueeze(1))
        
        Z = torch.cat(Z, dim=1)
        mus = torch.cat(mus, dim=1)
        sigmas = torch.cat(sigmas, dim=1)
            
            
        return Z[:, 1:, :], mus, sigmas
            
        
class MeanFieldCombiner(nn.Module): #MF
    def __init__(self, latent_dim, hidden_size):
        super(MeanFieldCombiner, self).__init__()
        
        self.latent_dim = latent_dim #100
        self.hidden_size = hidden_size #400
        
        self.linear_mu_right = torch.nn.Linear(self.hidden_size, 
                                                self.latent_dim)
        self.linear_sigma_right = torch.nn.Sequential( 
                                                torch.nn.Linear(self.hidden_size, 
                                                                self.latent_dim),
                                                torch.nn.Softplus())
        
        self.linear_mu_left = torch.nn.Linear(self.hidden_size,
                                                self.latent_dim)
        self.linear_sigma_left = torch.nn.Sequential(
                                                torch.nn.Linear(self.hidden_size, 
                                                                self.latent_dim),
                                                torch.nn.Softplus())
    
    def sample(self, mu, var): #epsilon ~ N(0, 1) * sqrt(var) + mean
        return mu + torch.sqrt(var) * torch.randn_like(mu)    
        
    def forward(self, h_right, h_left):
        bs, seq_len, _ = h_right.shape
        mus = []
        sigmas = []
        Z = []
        
        for t in range(seq_len):
            mu_r = self.linear_mu_right(h_right[:, t, :])
            var_r = self.linear_sigma_right(h_right[:, t, :])
            
            mu_l = self.linear_mu_left(h_left[:, t, :])
            var_l = self.linear_sigma_left(h_left[:, t, :])
            
            mu_t = (mu_r * var_l + mu_l * var_r) / (var_r + var_l)
            var_t = (var_r * var_l) / (var_r + var_l)
            
            Z.append(self.sample(mu_t, var_t).unsqueeze(1))
            mus.append(mu_t.unsqueeze(1))
            sigmas.append(var_t.unsqueeze(1))
            
            
        mus = torch.cat(mus, dim=1)
        sigmas = torch.cat(sigmas, dim=1)
        Z = torch.cat(Z, dim=1)
        
        return Z, mus, sigmas 
        
        

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=100, dropout=0.1, combiner_type='dks', rnn_type='rnn'):
        super(Inference, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.combiner_type = combiner_type
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size=input_dim, 
                                    hidden_size=hidden_dim, 
                                    dropout=self.dropout,
                                    bidirectional=True, 
                                    batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=input_dim, 
                                    hidden_size=hidden_dim,
                                    dropout=self.dropout, 
                                    bidirectional=True, 
                                    batch_first=True)
        else:
            raise NotImplementedError(f'rnn_type: {self.rnn_type} not implemented')
            

        if self.combiner_type == 'dks' or self.combiner_type == 'st-lr' or self.combiner_type == 'st-l':
            self.combiner = DKSCombiner(self.latent_dim, self.hidden_size)
            
        elif self.combiner_type == 'mf-lr':
            self.combiner = MeanFieldCombiner(self.latent_dim, self.hidden_size)
        
        else:
            raise NotImplementedError(f'combiner_type: {self.combiner_type} not implemented')

    def forward(self, x):
        out, hidden = self.rnn(x)
        self.h_left = out[:, :, :self.hidden_size]
        self.h_right = out[:, :, self.hidden_size:]
        
        if self.combiner_type == 'st-l':
            z = self.combiner(h_right=None ,h_left=self.h_left)
        
        elif self.combiner_type == 'st-lr':
            z= self.combiner(self.h_right, self.h_left)
        
        elif self.combiner_type == 'dks':
            z = self.combiner(self.h_right, h_left=None)
            
        elif self.combiner_type == 'mf-lr':
            z = self.combiner(self.h_right, self.h_left) 
        return z
        

class Generator(nn.Module):
    def __init__(self,  hidden_dim_em=100, 
                        latent_dim=100, 
                        hidden_dim_tr=200,
                        output_dim=88
                        ):
        super(Generator, self).__init__()
        self.hidden_dim_em = hidden_dim_em
        self.latent_dim = latent_dim
        self.hidden_dim_tr = hidden_dim_tr
        self.output_dim = output_dim
        
        self.emission = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, self.hidden_dim_em),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim_em, self.hidden_dim_em),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim_em, self.output_dim),
                    torch.nn.Sigmoid()
                )
        #transition function
        self.G = torch.nn.Sequential(#gating unit
                torch.nn.Linear(self.latent_dim, self.hidden_dim_tr), 
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim_tr, self.latent_dim), 
                torch.nn.Sigmoid()
            )

        self.H = torch.nn.Sequential(#proposed mean
                    torch.nn.Linear(self.latent_dim, self.hidden_dim_tr), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim_tr, self.latent_dim), 
                    torch.nn.Identity(),
                )
        self.mu_gated_linear = torch.nn.Linear(self.latent_dim, 
                                               self.latent_dim) #w_{mu_p} * z_{t-1} + b_{mu_p}
        
        self.sigma_gated_linear = torch.nn.Sequential( #w_{sigma_p} * relu(h_t) + b_{sigma_p}
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.latent_dim, self.latent_dim),
                                                    torch.nn.Softplus(),
                                                )
        
        # initialized W as identity matrix and b as zero vector
        for layer in self.mu_gated_linear.children():
            if isinstance(layer, nn.Linear):
                if layer.weight.data.size(0) == layer.weight.data.size(1):
                    nn.init.eye_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        
    def sample(self, mu, sigma):
        return mu + torch.sqrt(sigma) * torch.randn_like(mu)
    
    def get_mu_tr(self, z_t_1):
        """returns mu_t for a single sample depending on t-1"""
        
        g_out = self.G(z_t_1)
        one_minus_g = 1 - g_out
        mu_linear_out = self.mu_gated_linear(z_t_1)
        elementwise_mu_out = mu_linear_out * one_minus_g
        proposed_mean_out = self.H(z_t_1)
        mu_generator = (proposed_mean_out * g_out) + elementwise_mu_out
        return mu_generator
        
    def transition_forward(self, z_hat):
        batch_size, seq_len, latent_dim = z_hat.shape
        z_init = torch.randn(batch_size, 1, latent_dim, device=z_hat.device)  # Initial state

        # Lists to accumulate results for Z, mus, and sigmas
        Z_accum = [z_init]
        mus_accum = []
        sigmas_accum = []

        for t in range(seq_len):
            
            z_prev = Z_accum[-1].squeeze(1) #shape: (batch_size, latent_dim)
            mu_generator = self.get_mu_tr(z_prev)
            out = self.H(z_prev)
            sigma_generator = self.sigma_gated_linear(out)

            z_t = self.sample(mu_generator, sigma_generator).unsqueeze(1)  # Add sequence dimension
            Z_accum.append(z_t)  # Accumulate the result

            # Accumulate mus and sigmas
            mus_accum.append(mu_generator.unsqueeze(1))
            sigmas_accum.append(sigma_generator.unsqueeze(1))

        # Concatenate along sequence dimension
        Z = torch.cat(Z_accum, dim=1)
        mus = torch.cat(mus_accum, dim=1)
        sigmas = torch.cat(sigmas_accum, dim=1)

        return mus, sigmas
    
        

    def forward(self, z_hat):
        
        x_hat = self.emission(z_hat) #for reconstruction #shape: (batch_size, seq_len, output_dim)
        mus, sigmas = self.transition_forward(z_hat)

        return x_hat, mus, sigmas #x_hat for reconstruction, mus and sigmas for KL divergence
    
