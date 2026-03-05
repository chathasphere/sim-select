from torch.utils.data import Dataset


class Simulator(Dataset):
    def __init__(self, n_sample, mode=None):
        self.n_sample = n_sample
        self.data = None
        self.theta = None
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
    
    def simulate_data(self):
        raise NotImplementedError
    
    def get_observed_data(self):
        raise NotImplementedError
    
    def evaluate(self, posterior_params):
        pass