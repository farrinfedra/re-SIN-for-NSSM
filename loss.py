import torch
import numpy as np

def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m, v = self.enc.encode(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec.decode(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        kl = ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v)

        nelbo = kl + rec
        nelbo = torch.mean(nelbo, dim=0)
        kl = torch.mean(kl, dim=0)
        rec = torch.mean(rec, dim=0)

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec