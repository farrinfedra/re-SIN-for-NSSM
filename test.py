import torch 
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import midi_to_song, log_midis
from loss import kl_normal, log_bernoulli_with_logits
import logging
import torch.nn.functional as F
from dataloader import MusicDataset
from model import DVAE 
from einops import repeat, rearrange



def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml')
    argparser.add_argument('--device', type=str, default='mps')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--random', help='set true if sampling from random z', 
                            action='store_true', default=False)
    argparser.add_argument('--index', type=int, default=0)
    
    args = argparser.parse_args()
    return args

def main():
    args = get_arguments()
    config = OmegaConf.load(args.config)
    model = DVAE(input_dim=config.model.input_dim, 
                    hidden_dim=config.model.hidden_dim,
                    hidden_dim_em=config.model.hidden_dim_em, 
                    hidden_dim_tr=config.model.hidden_dim_tr, 
                    latent_dim=config.model.latent_dim).to(args.device)
    dataset = MusicDataset(config.dataset, split=config.sample.split)
    dataloader = DataLoader(dataset, batch_size=config.test.batch_size, shuffle=False)
    #load weights
    ckpt_path = config.test.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt)
    
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():   
        for j, (encodings, sequence_lengths) in enumerate(dataloader):
            encodings = encodings.to(args.device)
            x_hat, mus_inference, sigmas_inference, mus_generator, sigmas_generators = model(encodings)
            
            #get loss with only sum over latent dim dimension
            reconstruction_loss = log_bernoulli_with_logits(encodings, x_hat, sequence_lengths, T_reduction='none') 
            kl_loss = kl_normal(mus_inference, 
                                sigmas_inference, 
                                mus_generator, 
                                sigmas_generators, 
                                sequence_lengths,
                                T_reduction='none')
            
            kl_loss = kl_loss.sum(-1) #sum over T
            reconstruction_loss = reconstruction_loss.sum(-1) #sum over T
            
            #for a: #importance sampling
            z, mu_q, var_q = model.encoder(encodings)
            bs = encodings.shape[0]
            max_sequence_length = encodings.shape[1]
            loss_s = torch.zeros(bs)
            for s in range(config.test.S):
                z_s = mu_q + torch.sqrt(var_q) * torch.randn_like(mu_q)
                x_hat_s, mu_p, var_p = model.decoder(z_s)

                range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device) #shape: (batch, seq_len)
                mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
                mask = mask.to(sequence_lengths.device)
                mask = rearrange(mask, 'b s -> b s ()') #shape : (bs, seq_len, latent_dim)
                
                #binary cross entropy
                log_s_recosntruction_loss = log_bernoulli_with_logits(encodings, x_hat_s, sequence_lengths, T_reduction='sum')
                
                #gaussian log prob p(z)
                nll_p_z = F.gaussian_nll_loss(mu_p, z_s, var_p, reduction='none')
                nll_p_z = nll_p_z * mask.float()
                log_p_z = nll_p_z.sum(-1).sum(-1) #sum over latent dim and T #final shape (batch,)
                
                #gaussian log prob q(z|x)
                nll_q_z = F.gaussian_nll_loss(mu_q, z_s, var_q, reduction='none')
                nll_q_z = nll_q_z * mask.float()
                log_q_z = nll_q_z.sum(-1).sum(-1) #sum over latent dim and T #final shape (batch,)
                
                loss_s += torch.exp(-(log_s_recosntruction_loss + log_p_z - log_q_z))
            loss_s /= config.test.S
            loss_s = torch.log(loss_s)
            loss_s = -loss_s.mean(0)

            
            #for b:
            nelbo_matrix = reconstruction_loss + kl_loss
            nelbo_matrix = nelbo_matrix.sum(-1) #sum over batch_size
            sequence_lengths_sum = sequence_lengths.sum(-1)
            nelbo_b = nelbo_matrix / sequence_lengths_sum
            
            #for c:
            #divide each sample in batch by its sequence length
            nelbo_c = reconstruction_loss + kl_loss
            nelbo_c = nelbo_c / sequence_lengths.float()
            #take mean over batch
            nelbo_c = nelbo_c.mean(-1)
            
            # nelbo = nelbo.mean()

            
        if not args.debug:
            logging.info('=' * 50)
            logging.info(f'Validation Summary:\nEpoch: {epoch}, '
                        f'\nval_Iteration: {i}, '
                        f'\nval_NELBO: {nelbo.item()}, '
                        f'\nval_Reconstruction Loss: {reconstruction_loss.mean().item()}, '
                        f'\nval_KL Loss: {kl_loss.mean().item()}')
                
    avg_val_loss = val_epoch_loss / len(val_loader)