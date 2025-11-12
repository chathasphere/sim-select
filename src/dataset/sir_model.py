from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma

class SIRModel(Simulator):
    def __init__(self, beta, gamma, N, T, prior_scale, name=None, n_sample=None,
                 observed_seed=None):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.T = T
        self.obs_seed = observed_seed
        self.prior_scale = prior_scale
        self.n_sample = n_sample
        # TODO: save simulated data
        
        self.d_theta = 2
        self.d_x = 2 * T # asssume full observation for now
        self.data, self.theta = self.sample_model()
        
        
    def sample_model(self):
        thetas = self.sample_prior(self.n_sample, 7)
        xs = []
        for i in range(self.n_sample):
            random_seed = 7 * i # decorrelate random samples
            sim = self.simulate(
                thetas[i], random_seed
            )
            xs.append(sim)
        
        xs = torch.stack(xs).float()
            
        return xs, thetas.float()
        
    def sample_prior(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    def get_observed_data(self):
        pass
    
    def simulate(self, theta, seed):
        beta, gamma = theta # probably need to fix this
        N, T = self.N, self.T
        
        X = np.zeros((T, N))
        Y = np.zeros((T, N))
        
        # initialize infecteds
        n_init = int(0.02 * N)
        X[0][:n_init] = 1
        
        for t in range(1, T):
            S = (1 - X[t-1]) * (1 - Y[t-1])
            I = X[t-1] * (1 - Y[t-1])

            # simulate infections
            lam = beta * I.sum() / N
            p_i = 1 - np.exp(-lam)
            X[t] = np.where(S, np.random.binomial(1, p_i, N), X[t-1])
            
            # simulate recoveries
            p_r = 1 - np.exp(-gamma)
            Y[t] = np.where(I, np.random.binomial(1, p_r, N), Y[t-1])
            
        # summarize simulated data
        sX = X.sum(1)
        sY = Y.sum(1)
        
        # TODO: partial observation of data
        # TODO: compatibility with transformers?
        data = np.stack([sX, sY]).flatten()
        
        return torch.tensor(data).float()
    
    
    