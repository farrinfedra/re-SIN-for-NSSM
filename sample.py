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
    argparser.add_argument('--device', type=str, default='cpu')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--wandb', action='store_true', default=False)
    
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
    
    
    
    
    #sample
    model.eval()
    with torch.no_grad():
        z = torch.randn(config.sample.num_samples, config.model.latent_dim).to(args.device)
        x_hat, _, _ = model.decoder(z)
        x_hat = x_hat.squeeze().cpu().numpy()
        
        if config.sample.convert_to_song:
            midi_one_hot = latent_to_midi(x_hat)
            convert_to_song(x_hat, config.sample.save_dir)
        # print(x_hat.shape)
        # np.save('sample.npy', x_hat)
















if __name__ == '__main__':
    main()