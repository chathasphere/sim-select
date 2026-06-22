from torch.utils.data import DataLoader, Dataset, random_split
import torch
import lightning as L

class SimulatorModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, seed, batch_size, train_frac, task):
        super().__init__()
        self.dataset = dataset(mode=task)
        self.seed = seed
        self.batch_size = batch_size
        self.train_frac = train_frac
        if task not in ("criticism", "estimation"):
            raise ValueError("Invalid Training Task")
        self.task = task
        if self.task == "estimation":
            self.d_theta, self.d_x = self.dataset[0][0].shape, self.dataset[0][1].shape
        else:
            self.d_x = self.dataset[0].shape
            self.d_theta = None
        # assert len(self.d_x) == 2 # simulators should have output of shape (seq_len, features)

    
    def setup(self, stage):
        # TODO: run the sampling logic here.
        train_size = int(self.train_frac * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train, self.val = random_split(
                self.dataset,
                (train_size, val_size),
                torch.Generator().manual_seed(self.seed)
        )
        self.val_size = val_size
        
    
    def get_validation_set(self):
        return self.dataset[self.val.indices]
    


    def train_dataloader(self):
        train_batch_size = len(self.train) if self.batch_size is None else self.batch_size
        return DataLoader(self.train, train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        # load entire validation set as validation batch size
        return DataLoader(self.val, self.val_size)