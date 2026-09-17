from .sir_model import SIRModel
import torch
import numpy as np


class HetSIRModel(SIRModel):
    def __init__(self, betas, gamma, K, N, T, prior_scale, n_sample,
                 observed_seed=None, mode="estimation", noise=None, load_data=False):
        if type(betas) is list:
            assert K == len(betas)
            self.theta_true = np.append(betas, np.array(gamma))
            self.het = True
        elif (type(betas) is float) or (type(betas) is int):
            self.het = False
            self.theta_true = np.array([betas, gamma])
        self.K = K
        
        super().__init__(0, 0, N, T, prior_scale, n_sample,
                         observed_seed, mode, noise, load_data)
        
        
    def simulate(self, theta, seed=None):
        if self.het:
            beta = theta[:-1]
        else:
            beta = theta[0]
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
            if self.het:
                lam = beta[W] * I.sum() / N
            else:
                lam = beta * I.sum() / N
            p_i = 1 - np.exp(-lam)
            X[t] = np.where(S, X[t-1] + np.random.binomial(1, p_i, N), X[t-1])
            # simulate recoveries
            p_r = 1 - np.exp(-gamma)
            Y[t] = np.where(I, Y[t-1] + np.random.binomial(1, p_r, N), Y[t-1])
            

        # drop T=0 (it's fixed)   
        Y = Y[1:]
        finals = []
        # calculate removals in each susceptibility bucket
        # computing ~final size~ of epidemic in each bucket
        for k in range(K):
            finals.append(Y.T[W == k].mean(0)[-1])
        sY = np.array(finals)
        if self.mode == "criticism" and self.noiser:
            sY = sY + self.noiser(size=sY.shape) * (1 / self.N)
        return torch.tensor(sY).float()