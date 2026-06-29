from torch.utils.data import Dataset
import torch
from hydra.utils import get_original_cwd

class Simulator(Dataset):
    def __init__(self, n_sample, mode=None):
        self.n_sample = n_sample
        self.data = None
        self.theta = None
        self.theta_true = None
        self.mode = mode
        self.name = None
        self.load_data = False

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        if self.mode == "estimation":
            return self.theta[index], self.data[index]
        elif self.mode == "criticism":
            return self.data[index]
        else:
            raise ValueError("Invalid training task!")
    
    def simulate(self):
        raise NotImplementedError
    
    def sample_prior(self):
        raise NotImplementedError
    
    def get_observed_data(self):
        # generic enough that we can move this to parent class
        x_o = self.simulate(self.theta_true, self.obs_seed)
        return x_o.unsqueeze(0).float()
    
    def sample_model(self, load_data=False):
        save=False
        if load_data:
            try:
                self.load_data()
                return
            except FileNotFoundError:
                print("Saved simulations not found!")
                save = True
        print("Simulating samples...")
        self._sample_model(save=save)
    
    def _sample_model(self, save=False):
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
        
        self.data = torch.stack(ds).float()
        self.theta = thetas.float()
        if save:
            self.save_data(self.data, self.theta)
        return ds, thetas
    
    def load_data(self):
        prefix = get_original_cwd()
        self.data = torch.load(f"{prefix}/simulated_data/{self.name}_data_{self.n_sample}.pt")
        self.theta = torch.load(f"{prefix}/simulated_data/{self.name}_theta_{self.n_sample}.pt")
    
    
    def save_data(self, data, theta):
        prefix = get_original_cwd()
        torch.save(data, f"{prefix}/simulated_data/{self.name}_data_{self.n_sample}.pt")
        torch.save(theta, f"{prefix}/simulated_data/{self.name}_theta_{self.n_sample}.pt")