import torch 
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import midi_to_song, log_midis

import logging

from dataloader import MusicDataset
from model import DVAE 


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
            
            reconstruction_loss = log_bernoulli_with_logits(encodings, x_hat, sequence_lengths)
            kl_loss = kl_normal(mus_inference, 
                                sigmas_inference, 
                                mus_generator, 
                                sigmas_generators, 
                                sequence_lengths)
            
            nelbo = reconstruction_loss + kl_loss
            nelbo = nelbo.mean()
            val_epoch_loss += nelbo.item()
            
        if not args.debug:
            logging.info('=' * 50)
            logging.info(f'Validation Summary:\nEpoch: {epoch}, '
                        f'\nval_Iteration: {i}, '
                        f'\nval_NELBO: {nelbo.item()}, '
                        f'\nval_Reconstruction Loss: {reconstruction_loss.mean().item()}, '
                        f'\nval_KL Loss: {kl_loss.mean().item()}')
                
    avg_val_loss = val_epoch_loss / len(val_loader)