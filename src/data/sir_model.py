from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma
from src.utils import discrete_noiser



class SIRModel(Simulator):
    def __init__(self, beta, gamma, N, T, prior_scale, n_sample=None,
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None,
                 constant_hazard=0, initialize_samples=True):
        super().__init__(n_sample, mode)
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
        self.constant_hazard = constant_hazard
        self.theta_true = np.array([beta, gamma])
        if initialize_samples:
            self.data, self.theta = self.sample_model()
        else:
            self.data, self.theta = None, None
        
            
        # TODO: compatibility with transformers
        
        
    def sample_model(self):
        ds, thetas = super().sample_model()
        return ds, torch.log(thetas.float())
        
        
    def sample_prior(self, N, seed=None):
        if seed: torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    
    def simulate(self, theta, seed=None):
        beta, gamma = theta
        return self._simulate_sir(beta, gamma, 0, seed)

    
    
    def _simulate_sir(self, beta, gamma, delta, seed):
        
        N, T = self.N, self.T
        
        X = np.zeros((T+1, N))
        Y = np.zeros((T+1, N))
        A = np.zeros(N)
        
        # initialize infecteds
        n_init = int(0.02 * N)
        X[0][:n_init] = 1
        
        if seed is not None: np.random.seed(seed)
        
        for t in range(1, T+1):
            S = (1 - (X[t-1] - A)) * (1 - (Y[t-1] - A))
            I = (X[t-1] - A) * (1 - (Y[t-1] - A))
            R = Y[t-1] - A
            assert (S + I + R).sum() == N

            # simulate infections
            if self.constant_hazard:
                lam = beta / self.constant_hazard
            else:
                lam = beta * I.sum() / N
            p_i = 1 - np.exp(-lam)
            X[t] = np.where(S, X[t-1] + np.random.binomial(1, p_i, N), X[t-1])
            # simulate recoveries
            if gamma:
                p_r = 1 - np.exp(-gamma)
                Y[t] = np.where(I, Y[t-1] + np.random.binomial(1, p_r, N), Y[t-1])
            
            # simulate loss of immunity
            if delta:
                p_s = 1 - np.exp(-delta)
                A = np.where(R, A + np.random.binomial(1, p_s, N), A)
        
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
    
    
class SIRSModel(SIRModel):
    def __init__(self, beta, gamma, delta, N, T, prior_scale, n_sample=None,
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None):
        
        super().__init__(beta, gamma, N, T, prior_scale, n_sample,
                 observed_seed, partial_obs, mode, noise, initialize_samples=False)
        
        self.theta_true = np.array([beta, gamma, delta])
        self.data, self.theta = self.sample_model()
        
    def simulate(self, theta, seed=None):
        beta, gamma, delta =  theta
        return self._simulate_sir(beta, gamma, delta, seed)
        
       
class SIModel(SIRModel):
    def __init__(self, beta, N, T, prior_scale, n_sample=None,
                observed_seed=None, mode="estimation", noise=None):
    
        super().__init__(beta, 0, N, T, prior_scale, n_sample,
                    observed_seed, True, mode, noise, initialize_samples=False)
    
        self.theta_true = np.array([beta])
        self.data, self.theta = self.sample_model()
        
    def simulate(self, theta, seed=None):
        beta =  theta
        return self._simulate_sir(beta, 0, 0, seed)