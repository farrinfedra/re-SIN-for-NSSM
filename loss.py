from einops import repeat, rearrange
import torch
import torch.nn.functional as F

def kl_normal(qm, qv, pm, pv, sequence_lengths, T_reduction='mean'):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, sequence_dim, dim): q mean
        qv: tensor: (batch, sequence_dim, dim): q variance
        pm: tensor: (batch, sequence_dim, dim): p mean
        pv: tensor: (batch, sequence_dim, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    element_wise = element_wise.to(sequence_lengths.device)
    # mask out the padding after sequence length for each datapoint
    bs, max_sequence_length, _ = element_wise.shape

    # range_tensor = torch.arange(max_sequence_length).unsqueeze(0).expand(bs, -1) # [seq] -> [1, seq] -> [bs, seq]
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = mask.to(sequence_lengths.device)
    mask = rearrange(mask, 'b s -> b s ()')

    kl = element_wise * mask.float()
    kl = kl.sum(-1) #sum over latent dim

    if T_reduction == 'none':
        kl = kl
        
    elif T_reduction == 'mean':
        kl = kl.sum(-1) / sequence_lengths #mean over sequence length
        
    elif T_reduction == 'sum':
        kl = kl.sum(-1)

    
    return kl



def log_bernoulli_with_logits(x, logits, sequence_lengths, T_reduction='mean'):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = F.binary_cross_entropy(input=logits, target=x, reduction='none') #shape: (batch, seq_len, dim=88)
    # log_prob = F.binary_cross_entropy_with_logits(input=logits, target=x, reduction='none') #shape: (batch, seq_len, dim=88)
    bs, max_sequence_length, _ = x.shape
    
    range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device) #shape: (batch, seq_len)
    mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
    mask = mask.to(sequence_lengths.device)
    mask = rearrange(mask, 'b s -> b s ()')
    log_prob = log_prob.to(sequence_lengths.device)
    nll = log_prob * mask.float()
    
    #take sum over latent and mean of sequence lenghts
    nll = nll.sum(-1) #sum over latent dim
    # sum_T = sequence_lengths.float().sum(-1)
    if T_reduction == 'none':
        nll = nll
    elif T_reduction == 'mean':
        # nll = nll.mean(-1) #mean over sequence length
        nll = nll.sum(-1) / sequence_lengths
    elif T_reduction == 'sum':
        nll = nll.sum(-1)
    
    return nll


def importance_sampling(model, encodings, sequence_lengths, S):
    z, mu_q, var_q = model.encoder(encodings, sequence_lengths)
    bs = encodings.shape[0]
    max_sequence_length = encodings.shape[1]
    loss_s = torch.zeros(bs)
    all_exponent_args = []
    
    for _ in range(S):
        z_s = mu_q + torch.sqrt(var_q) * torch.randn_like(mu_q)
        x_hat_s, mu_p, var_p = model.decoder(z_s)

        range_tensor = repeat(torch.arange(max_sequence_length), 'l -> b l', b=bs).to(sequence_lengths.device) #shape: (batch, seq_len)
        mask = range_tensor < rearrange(sequence_lengths, 'b -> b ()')
        mask = mask.to(sequence_lengths.device)
        mask = rearrange(mask, 'b s -> b s ()') #shape : (bs, seq_len, latent_dim)
        
        #binary cross entropy
        log_s_recosntruction_loss = log_bernoulli_with_logits(encodings, x_hat_s, sequence_lengths, T_reduction='mean')
        
        #gaussian log prob p(z)
        nll_p_z = F.gaussian_nll_loss(mu_p, z_s, var_p, reduction='none')
        nll_p_z = nll_p_z * mask.float()
        log_p_z = nll_p_z.sum(-1).sum(-1) / sequence_lengths #sum over latent dim and T #final shape (batch,)
        
        #gaussian log prob q(z|x)
        nll_q_z = F.gaussian_nll_loss(mu_q, z_s, var_q, reduction='none')
        nll_q_z = nll_q_z * mask.float()
        log_q_z = nll_q_z.sum(-1).sum(-1) / sequence_lengths #sum over latent dim and T #final shape (batch,)
        
        # loss_s += torch.exp(-(log_s_recosntruction_loss + log_p_z - log_q_z))
        exponent_arg = -(log_s_recosntruction_loss + log_p_z - log_q_z)
        all_exponent_args.append(exponent_arg)
            
        
    loss_s = -torch.logsumexp(torch.stack(all_exponent_args), dim=0) + torch.log(torch.tensor(S, dtype=torch.float))
    loss_s = loss_s.mean()
    return loss_s