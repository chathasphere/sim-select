from torch.utils.data import Dataset
import torch

class Simulator(Dataset):
    def __init__(self, n_sample, mode=None):
        self.n_sample = n_sample
        self.data = None
        self.theta = None
        self.theta_true = None
        self.mode = mode

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        if self.mode == "estimation":
            return self.data[index], self.theta[index]
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
    
    def evaluate(self, posterior_params):
        pass