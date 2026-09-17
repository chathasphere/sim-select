from .simulator import Simulator
import torch
import numpy as np
from torch.distributions import Gamma
from itertools import chain
from numpy.random import exponential

class HouseholdEpidemicModel(Simulator):
    
    def __init__(self, H: list, prior_scale: list | float, n_sample: int, observed_data: list = None, 
                 mode="estimation", noise_scale=0.01, model_idx=0, load_data=False):
        
        self.prior_scale = prior_scale
        self.H = H # number of households by size
        self.observed = observed_data
        assert model_idx in (0, 1, 2)
        if model_idx == 0:
            assert type(prior_scale) is list
        else:
            assert type(prior_scale) is float
        self.model_idx = model_idx 
        
        
        super().__init__(n_sample, mode)
        
    
    def sample_prior(self, N, seed=None):
        if seed: torch.manual_seed(seed)
        prior = Gamma(1, 1 / torch.tensor(self.prior_scale))
        return prior.sample((N,))
    
    def sample_model(self):
        super().sample_model()
        self.theta = torch.log(self.theta)
    
    def get_observed_data(self):
        return torch.tensor(self.observed).unsqueeze(0)
    
    def simulate(self, theta, seed=None):
        if self.model_idx == 0:
            lam_G, lam_L = theta
        else:
            lam_G = theta
            lam_L = 0
        return self._simulate(lam_G, lam_L, seed)
    
    
    
    def _simulate(self, lam_G: float, lam_L: float, seed: int=None):
        # Sellke construction of household epidemic model
        

        K = list(range(1, len(self.H)+1)) # unique household sizes
        N = np.array(
            list(chain.from_iterable([[k] * self.H[k-1] for k in K]))
        )# size of each household, ordered

        m = len(N) # number of households

        S = N.copy()
        I = np.zeros(m)
        
        if self.model_idx == 1:
            lam_L = lam_G / sum(N)
        
        s = 0 # severity
        threshold = 0
        while threshold <= lam_G * s:
            # ith threshold indicates the additional global infectious pressure needed to trigger infection i+1
            k = np.random.choice(m, p=S/S.sum()) # randomly select a household with probabilities proportional to the number of susceptibles in each house
            h_i, h_s = self.simulate_household(S[k], lam_L, seed) # simulate an outbreak among remaining susceptibles within houehold k
            S[k] = S[k] - h_i # deplete susceptible(s)
            s += h_s
            I[k] = N[k] - S[k]

            if sum(S) > 0:
                threshold = threshold + exponential(sum(N) / sum(S)) # as the number of susceptibles depletes, the threshold gets higher?
            if sum(S) == 0:
                break

        result = []
        for i in range(1, len(self.H) +1):
            for j in range(i + 1):
                a = (I == j)[N==i].sum() / self.H[i-1]
                result.append(a)
        return torch.tensor(result)
        
    
    def simulate_household(self, n: int, lam_L: float, seed: int=None):
        # simulate a household epidemic in a household of size n
        # local infection rate lam_L
        # for simplicity we assume constant infectiousness periods
        if n == 1:
            i = 1
            s = 1
            # s = exponential(1)
        elif n > 1:
            T = self.set_thresholds(n, seed) # simulate list of local infection thresholds
            Q = np.ones(n)
            # Q = np.exponential(1, n)

            i = 0
            while True:
                i += 1
                s = sum(Q[:i])
                if i == n:
                    break
                    # everyone's infected
                elif T[i-1] > lam_L * s:
                    # the i-th threshold exceeds the cumulative force of infection from the first i infectives
                    # so the household epidemic terminates
                    break
        return i, s # return number of infecteds and severity
    
    def set_thresholds(self, n: int, seed: int=None):
    # set local thresholds for infectiousness in a household of size n
        if n == 1:
            return # thresholds not defined for a size 1 household
        if seed:
            np.random.seed(seed)
        thresh = np.zeros(n-1)
        thresh[0] = exponential(1/(n-1))
        if n > 2:
            for i in range(1, n-1):
                thresh[i] = thresh[i-1] + exponential(1 / (n-i)) # assume density-dependent transmission within household

        return thresh