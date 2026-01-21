import torch
import lightning as L
from zuko.flows.coupling import NICE
from zuko.flows.autoregressive import MAF
from zuko.flows.spline import NSF
from torch.nn import Module



# TODO: clip grad norm?
# TODO: default encoder: torch.nn.identity

class MarginalDensityFlow(L.LightningModule):
    # coupling flow with affine transformations
    def __init__(self, features: int, transforms: int, d_model: tuple[int, ...]=(64,64), 
                 lr: float =5e-4, weight_decay: float =0,
                 flow_type: str ="NSF", embedding: Module = None):
        super().__init__()
        
        if flow_type == "NSF":
            # TODO: make sure features are standardized before going into NSF
            self.flow = NSF(features=features, transforms=transforms, hidden_features=d_model) 
        elif flow_type == "MAF":
            self.flow = MAF(features=features, transforms=transforms, hidden_features=d_model)
        elif flow_type == "RealNVP":
            self.flow = NICE(features=features, transforms=transforms, hidden_features=d_model)
        
        self.lr = lr
        self.wd = weight_decay
        self.val_losses = []
        
        if embedding is None:
            self.embed = torch.nn.Identity
        
    def forward(self, x=None):
        # not really defined for a Zuko flow
        pass
    
    def sample(self, n):
        return self.flow().n_sample(n)
    
    def training_step(self, batch, batch_idx):
        x = self.embed(batch)
        loss = -self.flow().log_prob(x).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = self.embed(batch)
        loss = -self.flow().log_prob(x).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

