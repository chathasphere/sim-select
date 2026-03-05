from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma
from src.utils import discrete_noiser

class SIRModel(Simulator):
    def __init__(self, beta, gamma, N, T, prior_scale, n_sample=None,
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None,
                 constant_hazard=0, delta=0):
        super().__init__(n_sample, mode)
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.T = T
        self.obs_seed = observed_seed
        self.prior_scale = prior_scale
        self.partial = partial_obs
        # TODO: save simulated data
        if noise is not None:
            self.noiser = discrete_noiser(noise)
        else:
            self.noiser = None
        self.constant_hazard = constant_hazard
        self.delta = delta # defines reinfection rate
        
            
        # TODO: compatibility with transformers

        
        self.data, self.theta = self.sample_model()
        
        
    def sample_model(self):
        # consider making this a method of the parent class
        # the logic is pretty generic...
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
        if seed: torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    def get_observed_data(self):
        theta_true = np.array([self.beta, self.gamma])
        x_o = self.simulate(theta_true, self.obs_seed)
        return x_o.unsqueeze(0).float()
    
    def simulate(self, theta, seed=None):
        beta, gamma = theta
        N, T = self.N, self.T
        
        X = np.zeros((T+1, N))
        Y = np.zeros((T+1, N))
        
        # initialize infecteds
        n_init = int(0.02 * N)
        X[0][:n_init] = 1
        
        if seed is not None: np.random.seed(seed)
        
        for t in range(1, T+1):
            S = (1 - X[t-1]) * (1 - Y[t-1])
            I = X[t-1] * (1 - Y[t-1])

            # simulate infections
            if self.constant_hazard:
                lam = beta / self.constant_hazard
            else:
                lam = beta * I.sum() / N
            p_i = 1 - np.exp(-lam)
            X[t] = np.where(S, np.random.binomial(1, p_i, N), X[t-1])
            
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
    
    
    