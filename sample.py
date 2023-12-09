import torch 
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import midi_to_song, log_midis, collate_fn

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
                    latent_dim=config.model.latent_dim,
                    dropout=config.model.dropout,
                    combiner_type=config.model.combiner_type,
                    rnn_type=config.model.rnn_type).to(args.device)
    
    dataset = MusicDataset(config.dataset, split=config.sample.split)
    #load weights
    ckpt_path = config.sample.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt)
    
    os.makedirs(config.sample.exp_name, exist_ok=True)
    
    if args.random:
        z = torch.randn(config.sample.num_samples, 
                        config.sample.sequence_length, 
                        config.model.latent_dim).to(args.device)
        length = config.sample.sequence_length
        
    else:
        
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,  shuffle=False)
        x_orig = dataloader.dataset[args.index][0].to(args.device)
        z = x_orig.clone()
        z = z.unsqueeze(0) #shape: (1, seq_len, latent_dim)
        length = dataloader.dataset[args.index][1]
        sequence_length = torch.tensor(dataloader.dataset[args.index][1]).unsqueeze(0).to(args.device)
    #sample
    model.eval()
    with torch.no_grad():
        
        if args.random:
            x_hat, _, _ = model.decoder(z) 
        else:
            x_hat, _, _, _, _ = model(z, sequence_length) #x_hat shape: (bs, seq_len, 88)
        
        x_hat = x_hat.detach().cpu().numpy() 
        
        #trim the x_hat to the original length
        x_hat = x_hat[:, :length, :] #shape: (bs, length, 88)
        
        if args.random:
            for i in range(config.sample.num_samples):
                filename = f'random_sample_{i}'
                midi, orig_midi = dataset.recon_to_midi(x_hat=x_hat[i],
                                                         x_orig=None, 
                                                        threshhold=config.sample.threshhold) #orig_midi is none here
                midi_to_song(midi, filename, config.sample.exp_name)
                
        else: #sample from dataset
            x_hat = x_hat.squeeze(0) #shape: (seq_len, 88)
            filename = f'sample_{config.sample.split}_{args.index}_{config.sample.threshhold}'
            originial_filename = f'original_sample_{config.sample.split}_{args.index}'
            midi, orig_midi = dataset.recon_to_midi(x_hat, x_orig=x_orig, threshhold=config.sample.threshhold) #reconstruction to midi
            # print(f'midi: {midi}'), print(f'orig_midi: {orig_midi}')
            midi_to_song(midi, filename, config.sample.exp_name)
            midi_to_song(orig_midi, originial_filename, config.sample.exp_name)
        
        log_midis(midi, orig_midi)
        print('Done!')
                



if __name__ == '__main__':
    main()