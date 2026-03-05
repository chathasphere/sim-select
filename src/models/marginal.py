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
                d_model: tuple[int, ...]=(64,64), flow_type: str ="NSF", embedding: Module = None):
        super().__init__()
        
        if embedding is None:
            self.embed = torch.nn.Identity()
            try:
                features = prod(d_x)
            except TypeError:
                features = d_x
        else:
            self.embed = embedding(d_input = d_x)
            features = self.embed.d_model
        
        if flow_type == "NSF":
            self.flow = NSF(features=features, transforms=transforms, hidden_features=d_model) 
        elif flow_type == "MAF":
            self.flow = MAF(features=features, transforms=transforms, hidden_features=d_model)
        elif flow_type == "RealNVP":
            self.flow = NICE(features=features, transforms=transforms, hidden_features=d_model)
        
        self.optimizer = optimizer
        self.val_losses = []
        
        
    def forward(self, x=None):
        # not really defined for a Zuko flow
        pass
    
    def sample(self, n):
        return self.flow().sample((n,))
    
    def log_prob(self, x):
        return self.flow().log_prob(x)
    
    def training_step(self, batch, batch_idx=None):
        x = self.embed(batch)
        if len(x.shape) > 2: x = x.flatten(1)
        loss = -self.flow().log_prob(x).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx=None):
        x = self.embed(batch)
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
        x = self.embed(batch)
        # alternatively, return both 
        return x, self.log_prob(x)
