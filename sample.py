import torch 
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)

import wandb
import logging

from dataloader import MusicDataset
from model import DVAE 


def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml')
    argparser.add_argument('--device', type=str, default='mps')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--random', help='set true if sampling from random z', 
                            action='store_true', default=True)
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
    
    #load weights
    ckpt_path = config.sample.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    if args.random:
        z = torch.randn(config.sample.num_samples, 
                        config.sample.sequence_length, 
                        config.model.latent_dim).to(args.device)
        length = config.sample.sequence_length
        
    else:
        dataset = MusicDataset(config.dataset, split=config.sample.split)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        x_orig = dataloader.dataset[args.index][0].to(args.device)
        z = x_orig
        z = z.unsqueeze(0) #shape: (1, seq_len, latent_dim)
        length = dataloader.dataset[args.index][1]
    
    #sample
    model.eval()
    with torch.no_grad():
        
        if args.random:
            x_hat, _, _ = model.decoder(z) 
        else:
            x_hat, _, _, _, _ = model(z) #x_hat shape: (bs, seq_len, 88)
        
        x_hat = x_hat.detach().cpu().numpy() 
        
        #trim the x_hat to the original length
        x_hat = x_hat[:, :length, :] #shape: (bs, length, 88)
        
        if args.random:
            for i in range(config.sample.num_samples):
                filename = f'random_sample_{i}'
                midi, _ = dataset.recon_to_midi(x_hat[i], x_orig=None)
                midi_to_song(midi, filename)
                
        else: #sample from dataset
            filename = f'sample_{config.sample.split}_{args.index}'
            originial_filename = f'original_sample_{config.sample.split}_{args.index}'
            midi, orig_midi = dataset.recon_to_midi(x_hat[0], x_orig=x_orig) #reconstruction to midi
            midi_to_song(midi, filename)
            midi_to_song(orig_midi, originial_filename)
                







if __name__ == '__main__':
    main()