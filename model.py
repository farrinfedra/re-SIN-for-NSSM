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
        # Z = torch.zeros(h_right.shape[0], h_right.shape[1] + 1, self.latent_dim).to(h_right.device)
        # mus, sigmas = torch.zeros((b, seq_len, self.latent_dim)), \
                        # torch.ones((b, seq_len, self.latent_dim))
        
        mus = []
        sigmas = []
        z_init = torch.randn(b, 1, self.latent_dim).to(h_right.device) 
        Z = [z_init]
        
        for t in range(1, h_right.shape[1] + 1):
            z_prev = Z[-1].squeeze(1) #shape: (batch_size, latent_dim)
            
            h_combined = self.combiner(z_prev)
            h_combined = .5 * (F.tanh(h_combined) + h_right[:, t - 1, :])
            mu = self.mu_linear(h_combined) #shape: (batch_size, latent_dim)
            sigma = self.sigma_linear(h_combined)
            z_t = self.sample(mu, sigma)

            mus.append(mu.unsqueeze(1))
            sigmas.append(sigma.unsqueeze(1))
            Z.append(z_t.unsqueeze(1))
        
        Z = torch.cat(Z, dim=1)
        mus = torch.cat(mus, dim=1)
        sigmas = torch.cat(sigmas, dim=1)
            
            
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
        # self.z_0 = torch.zeros(, 1, self.hidden_size)
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
                    # torch.nn.Identity(),
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
        batch_size, seq_len, latent_dim = z_hat.shape
        z_init = torch.randn(batch_size, 1, latent_dim, device=z_hat.device)  # Initial state

        # Lists to accumulate results for Z, mus, and sigmas
        Z_accum = [z_init]
        mus_accum = []
        sigmas_accum = []

        for t in range(seq_len):
            
            z_prev = Z_accum[-1].squeeze(1)
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
        
        x_hat = self.emission(z_hat) #for reconstruction
        mus, sigmas = self.transition_forward(z_hat)

        return x_hat, mus, sigmas #x_hat for reconstruction, mus and sigmas for KL divergence
    
