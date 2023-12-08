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
from loss import kl_normal, log_bernoulli_with_logits

def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml')
    argparser.add_argument('--device', type=str, default='cpu')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--wandb', action='store_true', default=True)
    argparser.add_argument('--debug', action='store_true', default=False)
    
    args = argparser.parse_args()
    return args

def setup_logger():
    logging.basicConfig(filename='training_log_lstm.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')


def main():
    args = get_arguments()
    config = OmegaConf.load(args.config)
    
    tr_dataset = MusicDataset(config.dataset, split='train')
    val_dataset = MusicDataset(config.dataset, split='test')
    
    train_loader = DataLoader(tr_dataset, 
                            batch_size=config.train.batch_size, 
                            num_workers=config.train.num_workers,
                            pin_memory=True, #important for speed
                            shuffle=True)
    
    val_loader = DataLoader(val_dataset, 
                        batch_size=config.train.batch_size, 
                        num_workers=config.train.num_workers,
                        pin_memory=True)
                        # shuffle=True)
    
    device = torch.device(args.device)
    model = DVAE(input_dim=config.model.input_dim, 
                    hidden_dim=config.model.hidden_dim,
                    hidden_dim_em=config.model.hidden_dim_em, 
                    hidden_dim_tr=config.model.hidden_dim_tr, 
                    latent_dim=config.model.latent_dim,
                    combiner_type=config.model.combiner_type,
                    rnn_type=config.model.rnn_type).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    if not args.debug:
        setup_logger()

        
    if args.wandb:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(project=config.train.proj_name, 
                   entity=config.train.wandb_user_name,
                   config=config_dict)
        
    if not args.debug:    
        logging.info('Training Started')
        
    model.train()
    for epoch in range(config.train.epochs + 1):
        epoch_loss = 0
        for i, (encodings, sequence_lengths) in enumerate(train_loader):
            
            encodings = encodings.to(device)
            sequence_lengths = sequence_lengths.to(device)
            optimizer.zero_grad()
            
            x_hat, mus_inference, sigmas_inference, mus_generator, sigmas_generators = model(encodings)
            
            reconstruction_loss = log_bernoulli_with_logits(encodings, x_hat, sequence_lengths)
            kl_loss = kl_normal(mus_inference, 
                                sigmas_inference, 
                                mus_generator, 
                                sigmas_generators, 
                                sequence_lengths)
            
            
            nelbo = reconstruction_loss.to(device) + kl_loss.to(device)
            nelbo = nelbo.mean()
            epoch_loss += nelbo.item()
            
            nelbo.backward()
            optimizer.step()
            
            if not args.debug:
                logging.info('=' * 50)
                logging.info(f'Training Summary:\nEpoch: {epoch}, '
                            f'\nIteration: {i}, '
                            f'\nNELBO: {nelbo.item()}, '
                            f'\nReconstruction Loss: {reconstruction_loss.mean().item()}, '
                            f'\nKL Loss: {kl_loss.mean().item()}')
            
            
            if args.wandb:
                wandb.log({"tr_iteration": i, 
                            "tr_nelbo_mini_batch": nelbo.item(), 
                            "tr_recon_loss_mini_batch": reconstruction_loss.mean().item(), 
                            "tr_kl_loss_mini_batch": kl_loss.mean().item()})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # if args.wandb:
        #     wandb.log({"epoch": epoch,
        #                  "nelbo_epoch": epoch_loss,})
                        #  "recon_loss_epoch": reconstruction_loss.item(),
                        #  "kl_loss_epoch": kl_loss.item()})
            
        if config.train.save_model and epoch % config.train.save_every == 0:
            os.makedirs(config.train.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.train.save_dir, f'dvae_model_{epoch}.pt'))
        


    #validation
        if not args.debug:
            logging.info('Validation Started....')
            
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():   
            for j, (encodings, sequence_lengths) in enumerate(val_loader):
                encodings = encodings.to(device)
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
                if args.wandb:
                    wandb.log({"val_iteration": j, 
                                "val_nelbo_mini_batch": nelbo.item(), 
                                "val_recon_loss_mini_batch": reconstruction_loss.mean().item(), 
                                "val_kl_loss_mini_batch": kl_loss.mean().item()})
                    
        avg_val_loss = val_epoch_loss / len(val_loader)
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "avg_train_loss/epoch": avg_train_loss,
                "avg_val_loss/epoch": avg_val_loss
            })
        
    print(f"Finished Training for {config.train.epochs} epochs")
    
    if args.wandb:
        wandb.finish()




if __name__ == '__main__':
    main()