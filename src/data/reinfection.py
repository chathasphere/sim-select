from .simulator import Simulator
import torch
import numpy as np
import pandas as pd
from numpy.random import poisson
from hydra.utils import get_original_cwd

# basic SEIR model of influenza
# simulates the incidence (new infectious cases) time series

# TODO: read in the TDC influenza data. Return time series as the observed data, use it to set the initial conditions


class BaseReinfectionModel(Simulator):
    """SEIR model that returns incidence (new infectious cases) time series.

    Always returns only incidence as a tensor shaped `(1, T)`.
    """
    def __init__(self, init_I, init_S, beta, epsilon,
                 nu, rho, n_sample=None, mode="estimation", load_data=False, notebook=False):
        super().__init__(n_sample, mode)
        self.N = 284 # total population of TdC
        prefix = ".." if notebook else get_original_cwd()
        self.observed_data = pd.read_csv(f"{prefix}/data/fluTDC1971.csv")["obs"].values.astype(np.float32)
        # TODO: figure out a good way to resample the data to a lower time frequency
        # e.g. for timestep K days, take 
        # flu["obs"].rolling(K, min_periods=1).sum()[K-1::K]
        self.T = len(self.observed_data)
        assert init_I < self.N
        self.init_I = init_I
        assert init_S <= self.N - init_I
        self.init_S = init_S
        self.name = "reinfection-base"
        self.parameters = {
            "beta": (beta, torch.log), "epsilon": (epsilon, torch.log), "nu": (nu, torch.log), "rho": (rho, lambda x: torch.logit(x, eps=0.01))
        }
        # is there a good way to throw in some transformations here?
        self.data, self.theta = self.sample_model(load_data)
        
    def sample_model(self, load_data):
        ds, thetas = super().sample_model(load_data)
        transformed_thetas = torch.empty(thetas.shape)
        # TODO: put this into a designated method with its own inverse
        for i, k in enumerate(self.parameters):
            transform = self.parameters[k][1]
            transformed_thetas.T[i] = transform(thetas.T[i])
        return ds, transformed_thetas
            
    def get_observed_data(self):
        return torch.tensor(self.observed_data / self.N)

    def sample_prior(self, n_sample, seed=None):
        if seed: torch.manual_seed(seed)
        n_parameter = len(self.parameters)
        prior = torch.empty((n_parameter, n_sample))
        for i, k in enumerate(self.parameters):
            distr = self.parameters[k][0]
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
            tau = np.random.exponential(1.0 / total_rate)
            t_next = t + tau

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
        sInc = incidence / N


        data = sInc.reshape(1, -1)
        return torch.tensor(data).float()
