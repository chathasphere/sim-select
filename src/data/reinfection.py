from .simulator import Simulator
import torch
import numpy as np
import pandas as pd
from numpy.random import poisson
from hydra.utils import get_original_cwd

# basic SEIR model of influenza
# simulates the incidence (new infectious cases) time series

# TODO: logic to return the true incidence without applying the observational model
# could be useful for computing extinction probabilities

class BaseReinfectionModel(Simulator):
    """SEIR model that returns incidence (new infectious cases) time series.

    Always returns only incidence as a tensor shaped `(1, T)`.
    """
    def __init__(self, init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale=0.01, n_sample=None, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(n_sample, mode)
        self.N = 284 # total population of TdC
        prefix = ".." if notebook else get_original_cwd()
        self._observed_data = pd.read_csv(f"{prefix}/data/fluTDC1971.csv")["obs"].values.astype(np.float32)
        self.T = len(self._observed_data)
        assert init_I < self.N
        self.init_I = init_I
        assert init_S <= self.N - init_I
        self.init_S = init_S
        self.noise_scale = noise_scale
        self.name = "reinfection-base"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho}
        # rate parameters are constrained to the positive reals (P)
        # probability parameters are constrained to the unit interval (I)
        self.constraints = ("P", "P", "P", "I")
        self.transforms = {"P": torch.log, "I": lambda x: torch.logit(x, eps=0.01)}
        self.inverse_transforms = {"P": torch.exp, "I": torch.special.expit}
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
        assert prior.shape[1] == len(self.constraints)
        
        if inverse:
            # map back to natural domains
            inverted = torch.empty(prior.shape)
            for i, c in enumerate(self.constraints):
                t = self.inverse_transforms[c]
                inverted.T[i] = t(prior.T[i])
            return inverted
        else:
            # map constrained parameters to the real line
            # helps with neural posterior estimation
            transformed = torch.empty(prior.shape)
            for i, c in enumerate(self.constraints):
                t = self.transforms[c]
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
    
    @staticmethod
    def observation_model(incidence: np.array, rho: float):
        T = len(incidence)
        observed = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            observed[t] = poisson(rho * incidence[t])
        return observed
    
    def simulate(self, theta, seed=None, observed=True):
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
            
        
        # observed incidence counts
        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()
    
    
class Win(BaseReinfectionModel):
    """Reinfection model for the Window-of-reinfection hypothesis. Assumes that the acquisition of protective immunity
    is delayed after recovery, meaning that there's a window of susceptibility to reinfection by the same strain."""
    def __init__(self, init_I, init_S, beta, epsilon,
                 nu, rho,  gamma, tau, noise_scale=0.01, n_sample=None, mode="estimation", load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        
        self.name = "Win"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho, "gamma": gamma, "tau": tau}
        self.constraints = ("P", "P", "P", "I", "P", "P")
        # self.transforms = (torch.log, torch.log, torch.log, lambda x: torch.logit(x, eps=0.01), torch.log, torch.log)
        # self.inverse_transforms = (torch.exp, torch.exp, torch.exp, torch.special.expit, torch.exp, torch.exp)
        
        
    def simulate(self, theta, seed=None, observed=True):
        beta, epsilon, nu, rho, gamma, tau = theta
        N, T = self.N, self.T

        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = 0
        W = 0
        L = N - S - I # allow for some initial immune subjects
        states = (S, E, I, R, W, L)

        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E
            rate_inf = beta * S * I / N
            # incubation, E -> I
            rate_inc = epsilon * E
            # recovery, I -> R
            rate_rec = nu * I
            # window of reinfection, R -> W
            rate_win = gamma * R
            # reinfection, W -> E
            rate_reinf = beta * W * I / N
            # long term immunization, W -> L
            rate_imm = tau * W
            rates = (rate_inf, rate_inc, rate_rec, rate_win, rate_reinf, rate_imm)
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
                # reinfection window
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
        
        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()
    
    

class AoN(BaseReinfectionModel):
    def __init__(self, init_I, init_S, beta, epsilon, nu, rho, 
                 gamma, alpha, noise_scale=0.01, n_sample=None, mode="estimation",
                 load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        
        self.name = "AoN"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho, 
                           "alpha": alpha, "gamma": gamma}
        self.constraints = ("P", "P", "P", "I", "I", "P")
        
    def simulate(self, theta, seed=None, observed=True):
        beta, epsilon, nu, rho, alpha, gamma = theta
        N, T = self.N, self.T
        
        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = 0
        L = N - S - I # allow for some initial immune subjects
        states = (S, E, I, R, L)
        
        
        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E
            rate_inf = beta * S * I / N
            # incubation, E -> I
            rate_inc = epsilon * E
            # recovery, I -> R
            rate_rec = nu * I
            # return, R -> S
            rate_ret = (1 - alpha) * gamma * R
            # immunization, R -> L
            rate_imm = alpha * gamma * R
            rates = (rate_inf, rate_inc, rate_rec, rate_ret, rate_imm)
            total_rate = sum(rates)
            if total_rate <=0:
                break
            
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
                # return to susceptibility
                R -= 1
                S += 1
            else:
                # immunization
                R -= 1
                L += 1
            
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next
            
            
        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()

class PPI(BaseReinfectionModel):
    def __init__(self, init_I, init_S, beta, epsilon, nu, rho, sigma, gamma,
                 noise_scale=0.01, n_sample=None, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        
        self.name = "PPI"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho,
                           "sigma": sigma, "gamma": gamma}
        self.constraints = ("P", "P", "P", "I", "I", "P")
        
    def simulate(self, theta, seed=None, observed=True):
        beta, epsilon, nu, rho, sigma, gamma = theta
        N, T = self.N, self.T
        
        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = 0
        L = N - S - I # allow for some initial immune subjects
        states = (S, E, I, R, L)
        
        
        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E
            rate_inf = beta * S * I / N
            # incubation, E -> I
            rate_inc = epsilon * E
            # recovery, I -> R
            rate_rec = nu * I
            # reinfection, L -> E
            rate_reinf = sigma * beta * L * I / N
            # immunization, R -> L
            rate_imm = gamma * R
            rates = (rate_inf, rate_inc, rate_rec, rate_reinf, rate_imm)
            total_rate = sum(rates)
            if total_rate <=0:
                break
            
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
                # reinfection
                L -= 1
                E += 1
            else:
                # immunization
                R -= 1
                L += 1
            
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next

        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()
    
class InH(BaseReinfectionModel):
    def __init__(self, init_I, init_S, beta, epsilon, nu, rho, alpha, gamma,
                 noise_scale=0.01, n_sample=None, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        
        self.name = "InHI"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho,
                           "alpha": alpha, "gamma": gamma}
        self.constraints = ("P", "P", "P", "I", "I", "P")
        
    def simulate(self, theta, seed=None, observed=True):
        beta, epsilon, nu, rho, alpha, gamma = theta
        N, T = self.N, self.T
        
        # initial counts
        S = self.init_S
        E = 0
        I = self.init_I
        R = 0
        L = N - S - I # allow for some initial immune subjects
        states = (S, E, I, R, L)
        
        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)

        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E
            rate_inf = beta * S * I / N
            # incubation, E -> I
            rate_inc = epsilon * E
            # recovery, I -> R
            rate_rec = nu * I
            # reinfection, R -> I
            rate_reinf = (1 - alpha) * gamma * R
            # immunization, R -> L
            rate_imm = alpha * gamma * R
            rates = (rate_inf, rate_inc, rate_rec, rate_reinf, rate_imm)
            total_rate = sum(rates)
            if total_rate <=0:
                break
            
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
                # reinfection
                R -= 1
                I += 1
            else:
                # immunization
                R -= 1
                L += 1
            
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next

        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()
    
class TwoVi(BaseReinfectionModel):
    def __init__(self, init_I1, init_I2, init_S, beta_1, epsilon, nu, rho, gamma,
                 beta_2,
                 noise_scale=0.01, n_sample=None, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(init_I1, init_S, beta_1, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        self.init_I1 = init_I1
        self.init_I2 = init_I2
        self.name = "TwoVi"
        self.parameters = {"beta_1": beta_1, "epsilon": epsilon, "nu": nu, "rho": rho,
                        "gamma": gamma, "beta_2": beta_2}
        self.constraints = ("P", "P", "P", "I", "P", "P")
    
    def simulate(self, theta, seed=None, observed=True):
        beta_1, epsilon, nu, rho, gamma, beta_2 = theta
        N, T = self.N, self.T
        
        # initial counts
        S = self.init_S
        E1 = 0
        E2 = 0
        I1 = self.init_I1 # "low" infection group
        I2 = self.init_I2 # "high" infection group
        R1 = 0
        R2 = 0
        L1 = 0
        L2 = 0
        E12 = 0
        I12 = 0
        R12 = 0
        L12 = N - S - I1 - I2 # allow for some initial immune subjects
        
        states = (S, E1, E2, I1, I2, R1, R2, L1, L2, E12, I12, R12, L12)
        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)


        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E1 and S -> E2
            rate_inf1 = beta_1 * S * (I1 + I12) / N
            rate_inf2 = beta_2 * S * (I2 + I12) / N
            # incubation, E1 -> I1, E2 -> I2, E12 -> I12
            rate_inc1 = epsilon * E1
            rate_inc2 = epsilon * E2
            rate_inc12 = epsilon * E12
            # recovery, I1 -> R1, I2 -> R2, I12 -> R12
            rate_rec1 = nu * I1
            rate_rec2 = nu * I2
            rate_rec12 = nu * I12
            # immunization, R1 -> L1, R2 -> L2, and R12 -> L12
            rate_imm1 = gamma * R1
            rate_imm2 = gamma * R2
            rate_imm12 = gamma * R12
            # reinfection, L1 -> E12 and L2 -> E12
            rate_reinf1 = beta_2 * L1 * (I2 + I12) / N
            rate_reinf2 = beta_1 * L2 * (I1 * I12) / N

            rates = (rate_inf1, rate_inf2, rate_inc1, rate_inc2, rate_inc12, 
                     rate_rec1, rate_rec2, rate_rec12, rate_imm1, rate_imm2, 
                     rate_imm12, rate_reinf1, rate_reinf2)
            total_rate = sum(rates)
            
            
            if total_rate <=0:
                break
            
            d = np.random.exponential(1.0 / total_rate)
            t_next = t + d
    
                
            # choose event type
            draw = np.random.uniform(0.0, total_rate)
            if draw < sum(rates[:1]):
                # infection
                S -= 1
                E1 += 1
            elif draw < sum(rates[:2]):
                # incubation/progression
                S -= 1
                E2 += 1
            elif draw < sum(rates[:3]):
                E1 -= 1
                I1 += 1
                # add another case to the incidence for current time step
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:4]):
                E2 -= 1
                I2 += 1
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:5]):
                E12 -= 1
                I12 += 1
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:6]):
                I1 -= 1
                R1 += 1
            elif draw < sum(rates[:7]):
                I2 -= 1
                R2 += 1
            elif draw < sum(rates[:8]):
                I12 -= 1
                R12 += 1
            elif draw < sum(rates[:9]):
                R1 -= 1
                L1 += 1
            elif draw < sum(rates[:10]):
                R2 -= 1
                L2 += 1
            elif draw < sum(rates[:11]):
                R12 -= 1
                L12 += 1
            elif draw < sum(rates[:12]):
                L1 -= 1
                E12 += 1
            else:
                L2 -= 1
                E12 += 1
            
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next

        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()
    
    
class Mut(BaseReinfectionModel):
    def __init__(self, init_I, init_S, beta, epsilon, nu, rho, gamma,
                 sigma, mu, noise_scale=0.01, n_sample=None, mode="estimation", 
                 load_data=False, notebook=False):
        super().__init__(init_I, init_S, beta, epsilon,
                 nu, rho, noise_scale, n_sample, mode, load_data, notebook)
        self.init_I = init_I
        self.name = "Mut"
        self.parameters = {"beta": beta, "epsilon": epsilon, "nu": nu, "rho": rho,
                        "gamma": gamma, "sigma":sigma, "mu": mu}
        self.constraints = ("P", "P", "P", "I", "P", "I", "I")
    
    def simulate(self, theta, seed=None, observed=True):
        beta, epsilon, nu, rho, gamma, sigma, mu = theta
        N, T = self.N, self.T
        
        # initial counts
        S = self.init_S
        E1 = 0
        E2 = 0
        I1 = self.init_I
        I2 = 0
        R1 = 0
        R2 = 0
        L1 = 0
        L2 = 0
        E12 = 0
        I12 = 0
        R12 = 0
        L12 = N - S - I1 - I2 # allow for some initial immune subjects
        
        states = (S, E1, E2, I1, I2, R1, R2, L1, L2, E12, I12, R12, L12)
        if seed is not None:
            np.random.seed(seed)

        # true incidence counts per discrete time interval [t-1, t)
        incidence = np.zeros(T, dtype=np.float32)

        # mut_start, mut_stop = self.mutation_window
        # mutation_time = np.random.choice(np.arange(mut_start, mut_stop+1))
        mutation_time = np.floor(mu * 59)
        mutated = False

        t = 0
        while t < T and any(s > 0 for s in states[:-1]):
            # infection, S -> E1 and S -> E2
            rate_inf1 = beta * S * (I1 + I12) / N
            rate_inf2 = beta * S * (I2 + I12) / N
            # incubation, E1 -> I1, E2 -> I2, E12 -> I12
            rate_inc1 = epsilon * E1
            rate_inc2 = epsilon * E2
            rate_inc12 = epsilon * E12
            # recovery, I1 -> R1, I2 -> R2, I12 -> R12
            rate_rec1 = nu * I1
            rate_rec2 = nu * I2
            rate_rec12 = nu * I12
            # immunization, R1 -> L1, R2 -> L2, and R12 -> L12
            rate_imm1 = gamma * R1
            rate_imm2 = gamma * R2
            rate_imm12 = gamma * R12
            # reinfection, L1 -> E12 and L2 -> E12
            rate_reinf1 = sigma * beta * L1 * (I2 + I12) / N
            rate_reinf2 = sigma * beta * L2 * (I1 * I12) / N

            rates = (rate_inf1, rate_inf2, rate_inc1, rate_inc2, rate_inc12, 
                     rate_rec1, rate_rec2, rate_rec12, rate_imm1, rate_imm2, 
                     rate_imm12, rate_reinf1, rate_reinf2)
            total_rate = sum(rates)
            
            
            if total_rate <=0:
                break
            
            d = np.random.exponential(1.0 / total_rate)
            t_next = t + d
            
            if t_next >= mutation_time and not mutated:
                mutated = True
                if I1 > 0:
                    I1 -= 1
                    I2 += 1
                    t = t_next
                    continue

            # choose event type
            draw = np.random.uniform(0.0, total_rate)
            if draw < sum(rates[:1]):
                # infection
                S -= 1
                E1 += 1
            elif draw < sum(rates[:2]):
                # incubation/progression
                S -= 1
                E2 += 1
            elif draw < sum(rates[:3]):
                E1 -= 1
                I1 += 1
                # add another case to the incidence for current time step
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:4]):
                E2 -= 1
                I2 += 1
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:5]):
                E12 -= 1
                I12 += 1
                idx = int(np.floor(t_next))
                if idx < T:
                    incidence[idx] += 1.0
            elif draw < sum(rates[:6]):
                I1 -= 1
                R1 += 1
            elif draw < sum(rates[:7]):
                I2 -= 1
                R2 += 1
            elif draw < sum(rates[:8]):
                I12 -= 1
                R12 += 1
            elif draw < sum(rates[:9]):
                R1 -= 1
                L1 += 1
            elif draw < sum(rates[:10]):
                R2 -= 1
                L2 += 1
            elif draw < sum(rates[:11]):
                R12 -= 1
                L12 += 1
            elif draw < sum(rates[:12]):
                L1 -= 1
                E12 += 1
            else:
                L2 -= 1
                E12 += 1
            
            # if event occurs after the final observation window, stop
            if t_next >= T:
                break
            
            t = t_next

        if observed:
            data = self.observation_model(incidence, rho)
        else:
            data = incidence
            
        data = data / N
        if self.mode == "criticism":
            data = data + np.random.normal(scale=self.noise_scale, size=data.shape)

        return torch.tensor(data).float()