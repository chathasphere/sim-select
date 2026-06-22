from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma
from src.utils import discrete_noiser

# TODO: refactor this
# don't always need to supply a beta and gamma argument

class SIRModel(Simulator):
    def __init__(self, beta, gamma, N, T, prior_scale, n_sample,
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None,
                 constant_hazard=0, load_data=False, p_init=0.02, sample=True):
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
        self.name = "SIR"
        # self.data, self.theta = self.sample_model(load_data)
        # if load_data:
        #     self.save_data()
            
        # TODO: compatibility with transformers
        
        
    def sample_model(self, load_data):
        ds, thetas = super().sample_model(load_data)
        return ds, torch.log(thetas).float()
        
        
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
        n_init = int(self.p_init * N)
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
        if self.mode == "criticism" and self.noiser:
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
                 observed_seed=None, partial_obs=False, mode="estimation", noise=None,
                 load_data=False):
        
        super().__init__(beta, gamma, N, T, prior_scale, n_sample,
                 observed_seed, partial_obs, mode, noise, load_data)
        
        self.theta_true = np.array([beta, gamma, delta])
        
    def simulate(self, theta, seed=None):
        beta, gamma, delta =  theta
        self.name = "SIRS"
        return self._simulate_sir(beta, gamma, delta, seed)
        

class SIModel(SIRModel):
    def __init__(self, beta, N, T, prior_scale, n_sample=None,
                observed_seed=None, mode="estimation", noise=None,
                load_data=False):
    
        super().__init__(beta, 0, N, T, prior_scale, n_sample,
                    observed_seed, True, mode, noise, load_data)
    
        self.theta_true = np.array([beta])
        self.name = "SI"
        
    def simulate(self, theta, seed=None):
        beta =  theta
        return self._simulate_sir(beta, 0, 0, seed)
    
    
class HetSIRModel(SIRModel):
    def __init__(self, betas, gamma, N, T, prior_scale, n_sample,
                 observed_seed=None, mode="estimation", noise=None, load_data=False):
        self.K = len(betas)
        super().__init__(0, gamma, N, T, prior_scale, n_sample,
                         observed_seed, True, mode, noise, load_data)
        self.theta_true = np.append(betas, np.array(gamma))
        
    def simulate(self, theta, seed=None):
        beta = theta[:-1]
        gamma = theta[-1]
        return self._simulate_sir(beta, gamma, seed)
        
        
    def _simulate_sir(self, beta, gamma, seed):
        
        N, T = self.N, self.T
        K = self.K
        X = np.zeros((T+1, N))
        Y = np.zeros((T+1, N))
        W = np.arange(N) % K # susceptibility bins
        
        X[0][:K] = 1
        
        if seed is not None: np.random.seed(seed)
        
        for t in range(1, T+1):
            S = (1 - X[t-1]) * (1 - Y[t-1])
            I = X[t-1] * (1 - Y[t-1])
            R = Y[t-1]
            assert (S + I + R).sum() == N

            # simulate infections
            lam = beta[W] * I.sum() / N
            p_i = 1 - np.exp(-lam)
            X[t] = np.where(S, X[t-1] + np.random.binomial(1, p_i, N), X[t-1])
            # simulate recoveries
            p_r = 1 - np.exp(-gamma)
            Y[t] = np.where(I, Y[t-1] + np.random.binomial(1, p_r, N), Y[t-1])
            

        # drop T=0 (it's fixed)   
        X = X[1:]
        if self.mode == "criticism" and self.noiser:
            X = X + self.noiser(size=X.shape) * (1 / self.N) # could lead to floating point errors..?
        arrs = []
        # calculate incidence in each susceptibility bucket
        for k in range(K):
            arrs.append(X.T[W == k].mean(0))
        sX = np.concatenate(arrs)
        return torch.tensor(sX).float()
        

        