import torch
import lightning as L
from torch.utils.data import Dataset
import yaml
import numpy as np
import pandas as pd
import glob
from sklearn.datasets import make_moons
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig


    
def lower_tri(values, dim):
    if values.shape[0] > 1:
        L = torch.zeros(values.shape[0], dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[:, tril_ix[0], tril_ix[1]] = values
    # special case for non-batched inputs
    else:
        L = torch.zeros(dim, dim, device=values.device)
        tril_ix = torch.tril_indices(dim, dim)
        L[tril_ix[0], tril_ix[1]] = values[0]
    return L

def diag(values):
    if values.shape[0] > 1:
        L = torch.diag_embed(values)
    # special case for non-batched inputs
    else:
        L = torch.diag(values[0])
    return L


def save_results(posterior_params, val_losses, cfg, name):
    results = {"val_loss": val_losses[-1]}
    if posterior_params:
        # special case for homogeneous models
        if name in ["si-model", "crkp"]:
            mu = posterior_params[0].item()
            sigma = posterior_params[1].item()
            print(np.round(mu, 3))
            print(np.round(sigma, 3))
        else:
            mu = posterior_params[0].tolist()
            L = posterior_params[1]
            sigma = (L @ L.T).tolist()
            sdiag = (L @ L.T).diag().tolist()
            print(np.round(mu, 3))
            print(np.round(sdiag, 3)) # marginal variances
        results["mu"] = mu
        results["sigma"] = sigma
    for key in cfg["simulator"]:
        results[key] = cfg["simulator"][key]
    for key in cfg["model"]:
        results[key] = cfg["model"][key]
    for key in results:
        if type(results[key]) == ListConfig:
            results[key] = OmegaConf.to_object(results[key])
    with open("results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(results, yaml_file)
        
# reading multiruns

def get_results(path, multirun=True):
    extension =  "/results.yaml"
    if multirun: extension = "/**" + extension
    # if multirun:
    #     extension = "/**/results.yaml"
    # else:
    #     extension = "/results.yaml"
    results = glob.glob(path + extension)
    data = dict()
    for res in results:
        with open(res, "r") as stream:
            yml = yaml.safe_load(stream)
            for k, v in yml.items():
                if k not in data.keys():
                    data[k] = [v]
                else:
                    data[k].append(v)
    data = pd.DataFrame(data)
    # in practice, i don't tune these hyperparameters
    for c in ["_target_", "lr", "batch_size", "dropout", "seed"]:
        try:
            data.drop(columns = c, inplace=True)
        except KeyError:
            print(f"Missing column {c}")
    return data
        

class MoonsDataset(Dataset):
    def __init__(self, n_sample, random_state):
        self.n_sample = n_sample
        self.random_state = random_state
        self.data = self._make_data()

    def _make_data(self):
        arr = make_moons(self.n_sample, noise=0.05, random_state=self.random_state)[0]
        return torch.from_numpy(arr).float()

    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, index):
        return self.data[index]
