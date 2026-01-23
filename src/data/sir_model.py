from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma

class SIRModel(Simulator):
    def __init__(self, beta, gamma, N, T, prior_scale, name=None, n_sample=None,
                 observed_seed=None, partial_obs=False, mode="estimation"):
        super().__init__(n_sample, name, mode)
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.T = T
        self.obs_seed = observed_seed
        self.prior_scale = prior_scale
        # self.n_sample = n_sample
        self.partial = partial_obs
        # TODO: save simulated data
        
            
        # TODO: compatibility with transformers

        
        
        self.data, self.theta = self.sample_model()
        
        
    def sample_model(self):
        thetas = self.sample_prior(self.n_sample, 7)
        ds = [] # list of simulated data sets
        for i in range(self.n_sample):
            random_seed = 7 * i # decorrelate random samples
            sim = self.simulate(
                thetas[i], random_seed
            )
            ds.append(sim)
        
        ds = torch.stack(ds).float()
        
        # move parameters to the log scale
        return ds, torch.log(thetas.float())
        
    def sample_prior(self, N, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    def get_observed_data(self):
        theta_true = np.array([self.beta, self.gamma])
        x_o = self.simulate(theta_true, self.obs_seed)
        return x_o.unsqueeze(0).float()
    
    def simulate(self, theta, seed=None):
        beta, gamma = theta
        N, T = self.N, self.T
        
        X = np.zeros((T, N))
        Y = np.zeros((T, N))
        
        # initialize infecteds
        n_init = int(0.02 * N)
        X[0][:n_init] = 1
        
        if seed is not None: np.random.seed(seed)
        
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
        sX = X.mean(1)
        sY = Y.mean(1)
        
        
        if self.partial:
            data = sX # only incidence (new cases) is observed
            data = data.reshape(1, -1)
        else:
            data = np.stack([sX, sY])
        
        return torch.tensor(data).float()
    
    
    