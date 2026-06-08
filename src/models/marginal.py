import torch
import lightning as L
from zuko.flows.coupling import NICE
from zuko.flows.autoregressive import MAF
from zuko.flows.spline import NSF
from torch.nn import Module
from math import prod


# TODO: clean up API to match new hydra config setting
class MarginalDensityFlow(L.LightningModule):
    # coupling flow with affine transformations
    def __init__(self, d_x: int, transforms: int, optimizer:torch.optim.Optimizer,
                d_model: tuple[int, ...]=(64,64), flow_type: str ="NSF"):
        super().__init__()
        
        if type(d_x) is torch.Size:
            features = prod(d_x)
        else:
            features = d_x
        
        if flow_type == "NSF":
            self.flow = NSF(features=features, transforms=transforms, hidden_features=d_model) 
        elif flow_type == "MAF":
            self.flow = MAF(features=features, transforms=transforms, hidden_features=d_model)
        elif flow_type == "RealNVP":
            self.flow = NICE(features=features, transforms=transforms, hidden_features=d_model)
        
        self.optimizer = optimizer
        self.val_losses = []
        
        
    def forward(self, c=None):
        # not defined for a marginal flow
        pass
    
    def sample(self, n):
        return self.flow().sample((n,))
    
    def log_prob(self, x):
        return self.flow().log_prob(x)
    
    
    
    def training_step(self, batch, batch_idx=None):
        x = batch
        if len(x.shape) > 2: x = x.flatten(1)
        loss = -self.flow().log_prob(x).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx=None):
        x = batch
        if len(x.shape) > 2: x = x.flatten(1)
        loss = -self.flow().log_prob(x).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)
    
    def configure_optimizers(self):
        return self.optimizer(params = self.parameters())

    def predict_step(self, batch, batch_idx=None):
        x = batch
        # alternatively, return both 
        return x, self.log_prob(x)
