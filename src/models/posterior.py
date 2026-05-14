import torch
import lightning as L
from zuko.flows.coupling import NICE
from zuko.flows.autoregressive import MAF
from zuko.flows.spline import NSF
from torch.nn import Module
from math import prod


class NeuralPosteriorEstimator(L.LightningModule):
    def __init__(self, d_theta: int, d_x: int, transforms: int, optimizer:torch.optim.Optimizer,
                 d_model: tuple[int, ...]=(64,64), flow_type: str ="NSF", embedding: Module = None):
    
        super().__init__()
        
        if embedding is None:
            self.embed = torch.nn.Identity()
            if type(d_theta) is torch.Size:
                features = d_theta[0]
            if type(d_x) is torch.Size:
                context = prod(d_x)
        else:
            self.embed = embedding(d_input = d_x)
            context = self.embed.d_model
            
        if flow_type == "NSF":
            self.flow = NSF(features=features, context=context, transforms=transforms, hidden_features=d_model) 
        elif flow_type == "MAF":
            self.flow = MAF(features=features, context=context, transforms=transforms, hidden_features=d_model)
        elif flow_type == "RealNVP":
            self.flow = NICE(features=features, context=context, transforms=transforms, hidden_features=d_model)
        
        self.optimizer = optimizer
        self.val_losses = []
    
    
    def forward(self, c=None):
        # not really defined for a Zuko flow
        return self.flow(c)
    
    def sample(self, n, c):
        return self.flow(c).sample((n,))
    
    def log_prob(self, x, c):
        return self.flow(c).log_prob(x)
            
    def training_step(self, batch, batch_idx=None):
        x, c = batch
        c = self.embed(c)
        if len(c.shape) > 2: c = c.flatten(1)
        loss = -self.flow(c).log_prob(x).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx=None):
        x, c = batch
        c = self.embed(c)
        if len(c.shape) > 2: c = c.flatten(1)
        loss = -self.flow(c).log_prob(x).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)
    
    def configure_optimizers(self):
        return self.optimizer(params = self.parameters())

    def predict_step(self, batch, batch_idx=None):
        # TODO: figure out what makes sense here
        pass