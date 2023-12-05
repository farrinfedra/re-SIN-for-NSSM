import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import wandb

from dataloader import MusicDataset
from model import DVAE 

def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml')
    argparser.add_argument('--device', type=str, default='mps')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--wandb', action='store_true', default=False)
    
    args = argparser.parse_args()
    return args


def main():
    args = get_arguments()
    config = OmegaConf.load(args.config)
    
    dataset = MusicDataset(config.dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=config.train.batch_size, 
                            num_workers=config.train.num_workers,
                            shuffle=True)
    
    device = torch.device(args.device)
    model = DVAE(input_dim=config.model.input_dim, 
                    hidden_dim=config.model.hidden_dim,
                    hidden_dim_em=config.model.hidden_dim_em, 
                    hidden_dim_tr=config.model.hidden_dim_tr, 
                    latent_dim=config.model.latent_dim).to(device)
    
    if args.wandb:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(project=config.train.proj_name, 
                   entity=config.train.wandb_user_name,
                   config=config_dict)
        
        
    for i, (encodings, masks, sequence_lengths) in enumerate(dataloader):
        
        encodings = encodings.to(device)
        masks = masks.to(device)
        sequence_lengths = sequence_lengths.to(device)
        x_hat, mus_p_z, sigmas_p_z, mus_generator, sigmas_generators = model(encodings)
        
        #TODO: calculate loss
        
        print(f'encodings.shape: {encodings.shape}')
        print(f'masks.shape: {masks.shape}')
        print(f'sequence_lengths.shape: {sequence_lengths.shape}')
        
        # if args.wandb:
        #     wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        
        break

    if args.wandb:
        wandb.finish()




if __name__ == '__main__':
    main()