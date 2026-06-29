from .simulator import Simulator
import torch
import numpy as np
import pandas as pd
from numpy.random import poisson
from hydra.utils import get_original_cwd

# basic SEIR model of influenza
# simulates the incidence (new infectious cases) time series


class BaseReinfectionModel(Simulator):
    """SEIR model that returns incidence (new infectious cases) time series.

    Always returns only incidence as a tensor shaped `(1, T)`.
    """
    def __init__(self, init_I, init_S, beta, epsilon,
                 nu, rho, n_sample=None, noise_scale=0.01, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(n_sample, mode)
        self.N = 284 # total population of TdC
        prefix = ".." if notebook else get_original_cwd()
        self._observed_data = pd.read_csv(f"{prefix}/data/fluTDC1971.csv")["obs"].values.astype(np.float32)
        # TODO: figure out a good way to resample the data to a lower time frequency
        # e.g. for timestep K days, take 
        # flu["obs"].rolling(K, min_periods=1).sum()[K-1::K]
        self.T = len(self._observed_data)
        assert init_I < self.N
        self.init_I = init_I
        assert init_S <= self.N - init_I
        self.init_S = init_S
        self.noise_scale = noise_scale
        self.name = "reinfection-base"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho}
        self.transforms = (torch.log, torch.log, torch.log, lambda x: torch.logit(x, eps=0.01))
        self.inverse_transforms = (torch.exp, torch.exp, torch.exp, torch.special.expit)
        self.load_data = load_data
        self.original_data = None
        
    def sample_model(self):
        super().sample_model(self.load_data)
        self.theta = self.transform_parameters(self.theta)
        
    def resample(self, k:int, pad:int=60):
        if k == 1:
            # return to original length
            if self.original_data is not None: self.data = self.original_data
            self.original_data = None
        else:
            # prevent resampling from occurring twice
            assert self.original_data is None
            M, T = self.data.shape
            y = torch.zeros((M, pad))
            y[:, :T] = self.data
            self.original_data = self.data.clone()
            self.data = y.unfold(1, k, k).sum(-1)
        
    # def resample(self, x: torch.tensor, k: int, pad=60):
    #     M, T = x.shape
    #     assert T == self.T
    #     y = torch.zeros((M, pad))
    #     y[:, :T] = x
    #     return y.unfold(1, k, k).sum(-1)
        
    
    def transform_parameters(self, prior:torch.tensor, inverse=False):
        # assumes prior is of shape (N, 4)
        # N samples of 4 components
        assert prior.shape[1] == len(self.transforms) == len(self.inverse_transforms)
        
        if inverse:
            # map back to natural domains
            inverted = torch.empty(prior.shape)
            for i, t in enumerate(self.inverse_transforms):
                inverted.T[i] = t(prior.T[i])
            return inverted
        else:
            # map constrained parameters to the real line
            # helps with neural posterior estimation
            transformed = torch.empty(prior.shape)
            for i, t in enumerate(self.transforms):
                transformed.T[i] = t(prior.T[i])
            return transformed
            
            

    def get_observed_data(self):
        return torch.tensor(self._observed_data / self.N).unsqueeze(0)

    def sample_prior(self, n_sample, seed=None):
        if seed: torch.manual_seed(seed)
        n_parameter = len(self.parameters)
        prior = torch.empty((n_parameter, n_sample))
        for i, k in enumerate(self.parameters):
            distr = self.parameters[k]
            prior[i] = distr.sample((n_sample,))
        return prior.T
    
    def simulate(self, theta, seed=None):
        beta, epsilon, nu, rho = theta
        N, T = self.N, self.T

        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = N - S - I # allow for some initial immune subjects

        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)
        # observed incidence counts
        observed = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and (S > 0 or E > 0 or I > 0):
            rate_inf = beta * S * I / N
            rate_inc = epsilon * E
            rate_rec = nu * I
            total_rate = rate_inf + rate_inc + rate_rec
            if total_rate <= 0:
                break

            # time to next event
            d = np.random.exponential(1.0 / total_rate)
            t_next = t + d

            # choose event type
            draw = np.random.uniform(0.0, total_rate)
            if draw < rate_inf:
                event = "infection"
            elif draw < rate_inf + rate_inc:
                event = "progression"
            else:
                event = "recovery"

            # if event occurs after the final observation window, stop
            if t_next >= T:
                break

            # record E->I events as incidence in the appropriate discrete bin
            if event == "infection":
                S -= 1
                E += 1
            elif event == "progression":
                E -= 1
                I += 1
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            else:
                I -= 1
                R += 1

            t = t_next
            
        # observational model
        # maybe make this optional if class arg observation_model=False or smth
        for t in range(1, T):
            observed[t] = poisson(rho * incidence[t])
            
        
        # incidence as proportion of population per time step
        data = incidence / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        # if we want to use transformer embeddings we need to put in another dimension here
        # data = data.reshape(1, -1)
        return torch.tensor(data).float()
    
    
class Window(BaseReinfectionModel):
    """Reinfection model for the Window-of-reinfection hypothesis. Assumes that the acquisition of protective immunity
    is delayed after recovery, meaning that there's a window of susceptibility to reinfection by the same strain."""
    def __init__(self, init_I, init_S, beta, epsilon,
                 nu, rho, gamma, tau, n_sample=None, mode="estimation", load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, n_sample, mode, load_data, notebook)
        
        self.name = "win"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho, "gamma": gamma, "tau": tau}
        self.transforms = (torch.log, torch.log, torch.log, lambda x: torch.logit(x, eps=0.01), torch.log, torch.log)
        self.inverse_transforms = (torch.exp, torch.exp, torch.exp, torch.special.expit, torch.exp, torch.exp)
        
        
    def simulate(self, theta, seed=None):
        beta, epsilon, nu, rho, gamma, tau = theta
        N, T = self.N, self.T

        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = 0
        W = 0
        L = N - S - I # allow for some initial immune subjects

        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)
        # observed incidence counts
        observed = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and (S > 0 or E > 0 or I > 0 or R > 0 or W > 0):
            rate_inf = beta * S * I / N
            rate_inc = epsilon * E
            rate_rec = nu * I
            # reentry into the transmission process
            rate_reentry = gamma * R
            rate_reinf = beta * W * I / N
            rate_imm = tau * W
            rates = (rate_inf, rate_inc, rate_rec, rate_reentry, rate_reinf, rate_imm)
            total_rate = sum(rates)
            if total_rate <= 0:
                break

            # time to next event
            d = np.random.exponential(1.0 / total_rate)
            t_next = t + d

            # choose event type
            draw = np.random.uniform(0.0, total_rate)
            if draw < sum(rates[:1]):
                # infection
                S -= 1
                E += 1
            elif draw < sum(rates[:2]):
                # incubation/progression
                E -= 1
                I += 1
                # add another case to the incidence for current time step
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:3]):
                # recovery
                I -= 1
                R += 1
            elif draw < sum(rates[:4]):
                # reentry
                R -= 1
                W += 1
            elif draw < sum(rates[:5]):
                # reinfection
                W -= 1
                E += 1
            else:
                # immunization
                W -= 1
                L += 1
        
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next
        # observational model
        # maybe make this optional if class arg observation_model=False or smth
        for t in range(1, T):
            observed[t] = poisson(rho * incidence[t])
            
        
        # incidence as proportion of population per time step
        sInc = incidence / N


        data = sInc.reshape(1, -1)
        return torch.tensor(data).float()
    
    
