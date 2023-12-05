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
                    latent_dim=100
                    ):
        super(DVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_em = hidden_dim_em
        self.hidden_dim_tr = hidden_dim_tr
        self.latent_dim = latent_dim
        self.output_dim = input_dim
        
        self.encoder = Inference(self.input_dim, 
                                self.hidden_dim, 
                                self.latent_dim)
        self.decoder = Generator(self.hidden_dim_em, 
                                self.latent_dim, 
                                self.hidden_dim_tr, 
                                self.output_dim)

    def forward(self, x):
        z, mus_inference, sigmas_inference = self.encoder(x)
        x_hat, mus_generator, sigmas_generator = self.decoder(z)
        return x_hat, mus_inference, sigmas_inference, mus_generator, sigmas_generator


class Combiner(nn.Module): #DKS
    def __init__(self, latent_dim, hidden_size):
        super(Combiner, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        
        # print(f'latent_dim: {latent_dim}, hidden_size: {hidden_size}')
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
        
    def sample(self, mu, sigma):
        return mu + sigma * torch.randn_like(mu)
        

    def forward(self, h_right):
        #shape Z: (batch_size, seq_len, latent_dim)
        b, seq_len, _ = h_right.shape
        Z = torch.zeros(h_right.shape[0], h_right.shape[1] + 1, self.latent_dim) 
        mus, sigmas = torch.zeros((b, seq_len, self.latent_dim)), \
                        torch.ones((b, seq_len, self.latent_dim))
        
        for t in range(1, h_right.shape[1] + 1):
            # print(f'Z[:, t - 1, :].shape: {Z[:, t - 1, :].shape}')
            h_combined = self.combiner(Z[:, t - 1, :])
            h_combined = .5 * (F.tanh(h_combined) + h_right[:, t - 1, :])
            mu = self.mu_linear(h_combined)
            sigma = self.sigma_linear(h_combined)
            z_t = self.sample(mu, sigma)
            
            mus[:, t - 1, :] = mu #TODO: check
            sigmas[:, t - 1, :] = sigma #TODO: check
            Z[:, t, :] = z_t
            
            
        return Z[:, 1:, :], mus, sigmas
            
        

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=100):
        super(Inference, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        self.rnn = torch.nn.RNN(input_size=input_dim, 
                                hidden_size=hidden_dim, 
                                bidirectional=True, 
                                batch_first=True)
        self.z_0 = torch.zeros(1, 1, self.hidden_size)
        # self.h_right = None
        # self.h_left = None
        self.combiner = Combiner(self.latent_dim, self.hidden_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        self.h_right = out[:, :, :self.hidden_size]
        self.h_left = out[:, :, self.hidden_size:]
        z = self.combiner(self.h_right)
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
        return mu + sigma * torch.randn_like(mu)
    
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
        
        Z = torch.zeros_like(z_hat) #shape: (batch_size, seq_len, latent_dim)
        shape = Z.shape
        z_init = torch.randn_like(z_hat[:, 0, :]) #shape: (batch_size, latent_dim)
        # Z = torch.cat((z_init, z_hat), dim=1)
        mus, sigmas = torch.zeros(shape), torch.ones(shape)
        
        Z[:, 0, :] = z_init
        for t in range(1, Z.shape[1]):
            mu_generator = self.get_mu_tr(Z[:, t - 1, :])
            sigma_generator = self.sigma_gated_linear(self.H(Z[:, t - 1, :]))
            Z[:, t, :] = self.sample(mu_generator, sigma_generator) #z_t = mu_t + sigma_t * epsilon_t
            
            mus[:, t, :] = mu_generator #mu_t
            sigmas[:, t, :] = sigma_generator #sigma_t
            
        return mus, sigmas #contain mu_1=0 and sigma_1=1 for KL divergence
        

    def forward(self, z_hat):
        
        x_hat = self.emission(z_hat) #for reconstruction
        mus, sigmas = self.transition_forward(z_hat)

        return x_hat, mus, sigmas #x_hat for reconstruction, mus and sigmas for KL divergence
    
