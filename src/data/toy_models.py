from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Normal
import math

class NormalNormal(Simulator):
    def __init__(self, N, prior_mu, prior_sigma, observed_seed, 
                 dim, n_sample, true_sigma, false_sigma=None, mode="criticism"):
        super().__init__(n_sample, mode)
        self.N = N
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.d = dim
        self.obs_seed = observed_seed
        self.true_sigma = true_sigma
        self.false_sigma = false_sigma if false_sigma else true_sigma

        self.data, self.theta = self.sample_model()
        
    def sample_model(self):
        thetas = self.sample_prior(self.n_sample, 10)
        ds = []
        for i in range(self.n_sample):
            random_seed = 7 * i # decorrelate random samples
            sim = self.simulate(
                thetas[i], random_seed
            )
            ds.append(sim)
        
        ds = torch.stack(ds).float()
        # standardize the data?
        return ds, thetas.float()
    
    def sample_prior(self, N=None, seed=None):
        if seed: torch.manual_seed(seed)
        prior = Normal(self.prior_mu, self.prior_sigma)
        if N:
            return prior.sample((N, self.d))
        else:
            return prior.sample((self.d,))
    
    def simulate(self, theta, seed=None, sigma=None):
        if seed: torch.manual_seed(seed)
        if sigma:
            likelihood = Normal(theta, sigma)
        else:
            likelihood = Normal(theta, self.false_sigma)
        xs = likelihood.sample((self.N,))
        if self.N > 1:
            # summarize
            return torch.cat((xs.mean(0), xs.var(0)))
        else:
            return xs
    
    def get_observed_data(self):
        theta_true = self.sample_prior(seed=self.obs_seed)
        x_o = self.simulate(theta_true, self.obs_seed, self.true_sigma)
        return x_o.unsqueeze(0).float()
        


class ConditionalMoonsDataset(Simulator):
    # test data set for conditional normalizing flows
    def __init__(self, n_sample):
        super().__init__(n_sample, "estimation")
        self.data, self.theta = self.sample_model()
         
    def sample_model(self):
        np.random.seed(8)
        theta = np.random.uniform(-1, 1, (2, self.n_sample))
        a = np.random.uniform(low=-math.pi/2, high=math.pi/2, size=self.n_sample)
        r = np.random.normal(loc=0.1, scale=0.01, size=self.n_sample)
        p = np.stack([r * np.cos(a) + 0.25, r * np.sin(a)])
        b0 = - np.abs(theta[0] + theta[1]) / math.sqrt(2)
        b1 = (-theta[0] + theta[1]) / math.sqrt(2)
        x = np.stack([p[0] + b0, p[1] + b1])
        return torch.tensor(x.T).float(), torch.tensor(theta.T).float()
    
    def get_observed_data(self):
        return torch.tensor([0,0]).unsqueeze(0).float()