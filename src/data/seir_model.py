from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma
from src.utils import discrete_noiser

class SEIRModel(Simulator):
    def __init__(self, beta, sigma, gamma, N, T, prior_scale, n_sample=None,
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None,
                 load_data=False):
        super().__init__(n_sample, mode)
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        self.T = T
        self.obs_seed = observed_seed
        self.prior_scale = prior_scale
        self.partial = partial_obs
        # TODO: save simulated data
        if noise is not None:
            self.noiser = discrete_noiser(**noise)
        else:
            self.noiser = None
        self.name = "SEIR"
        self.theta_true = np.array([beta, sigma, gamma])
        # self.data, self.theta = self.sample_model(load_data)
        
        
    def sample_model(self, load_data):
        ds, thetas = super().sample_model(load_data)
        # consider moving thetas to the log scale
        return ds, thetas.float()
        
    def sample_prior(self, N, seed=None):
        if seed: torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    def simulate(self, theta, seed=None):
        beta, sigma, gamma = theta
        N, T = self.N, self.T
        
        X = np.zeros((T+1, N))
        Y = np.zeros((T+1, N))
        W = np.zeros(N)
        
        # initialize infecteds
        n_init = int(0.02 * N)
        X[0][:n_init] = 1
        
        if seed is not None: np.random.seed(seed)
        
        for t in range(1, T+1):
            S = (1 - W) * (1 - X[t-1]) * (1 - Y[t-1])
            E = W * (1 - X[t-1]) * (1 - Y[t-1])
            I = X[t-1] * (1 - Y[t-1])
            R = Y[t-1]
            assert (S + E + I + R).sum() == N

            # simulate exposure
            lam = beta * I.sum() / N
            p_e = 1 - np.exp(-lam)
            W = np.where(S, np.random.binomial(1, p_e, N), W)
            
            # simulate infections
            p_i = 1 - np.exp(-sigma)
            X[t] = np.where(E, np.random.binomial(1, p_i, N), X[t-1])
            
            # simulate recoveries
            p_r = 1 - np.exp(-gamma)
            Y[t] = np.where(I, np.random.binomial(1, p_r, N), Y[t-1])
            
        
        # drop T=0 (it's fixed)   
        X = X[1:]
        Y = Y[1:]
            
        # summarize simulated data
        sX = X.mean(1)
        sY = Y.mean(1)
        
        # noising stage for density estimation
        if self.mode == "criticism" and self.noiser is not None:
            sX = sX + self.noiser(size=sX.shape) * (1 / self.N)
            sY = sY + self.noiser(size=sX.shape) * (1 / self.N)
        
        if self.partial:
            data = sX # only incidence (new cases) is observed
            data = data.reshape(1, -1)
        else:
            data = np.stack([sX, sY])
        
        return torch.tensor(data).float()
    
    
    