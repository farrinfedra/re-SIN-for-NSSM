import torch 
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import midi_to_song, log_midis
from loss import kl_normal, log_bernoulli_with_logits, importance_sampling
import logging
import torch.nn.functional as F
from dataloader import MusicDataset
from model import DVAE 
from einops import repeat, rearrange


def setup_logger(config, rand):
    filename = 'testinggggg.log'
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml')
    argparser.add_argument('--device', type=str, default='mps')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--random', help='set true if sampling from random z', 
                            action='store_true', default=False)
    
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
    dataloader = DataLoader(dataset, batch_size=config.test.batch_size, shuffle=False)
    #load weights
    ckpt_path = config.test.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt)
    
    setup_logger(config, rand=0)
    
    model.eval()

    a = 0
    b = 0
    c = 0
    total_nelbo_b = 0
    total_sequence_lengths_sum = 0
    total_nelbo_c = 0
    total_count = 0
    
    with torch.no_grad():   
        for j, (encodings, sequence_lengths) in enumerate(dataloader):
            
            encodings = encodings.to(args.device)
            sequence_lengths = sequence_lengths.to(args.device)
            
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
            loss_s = importance_sampling(model, encodings, sequence_lengths, S=config.test.S)
            a += loss_s
            
            #for b:
            nelbo_matrix = reconstruction_loss + kl_loss
            nelbo_matrix = nelbo_matrix.sum(-1) #sum over batch_size
            sequence_lengths_sum = sequence_lengths.sum(-1)
            nelbo_b = nelbo_matrix / sequence_lengths_sum
            b += nelbo_b
            
            
            #for c:
            #divide each sample in batch by its sequence length
            nelbo_c = reconstruction_loss + kl_loss
            nelbo_c = nelbo_c / sequence_lengths.float()
            #take mean over batch
            nelbo_c = nelbo_c.mean(-1)
            c += nelbo_c
            
       
            logging.info('=' * 50)
            logging.info(
                f'Testing Summary:\n'
                f'Iteration: {j}, '
                f'\na: {loss_s.item()}, '
                f'\n(b): {nelbo_b.item()}, '
                f'\nc: {nelbo_c.item()}'
            )
    
    total_nelbo_b += nelbo_matrix.sum().item()
    total_sequence_lengths_sum += sequence_lengths.sum().item()
    
    total_nelbo_c += nelbo_c.sum().item()
    total_count += sequence_lengths.size(0)
    
    final_b = total_nelbo_b / total_sequence_lengths_sum
    final_c = total_nelbo_c / total_count
    
    # logging.info('=' * 50)
    # logging.info(
    #     f'Final Testing Summary:\n'
    #     f'a: {a.item() / len(dataloader)}, '
    #     f'\n(b): {b.item() / len(dataloader)}, '
    #     f'\nc: {c.item() / len(dataloader)}'
    # )
    logging.info('=' * 50)
    logging.info(
        f'Final Testing Summary:\n'
        f'a: {a.item() / len(dataloader)}, '
        f'\n(b): {final_b}, '
        f'\nc: {final_c}'
    )
    logging.info('=' * 50)
    
    
if __name__ == '__main__':
    main()