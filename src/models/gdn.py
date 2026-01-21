import lightning as L
import torch
from torch.nn import Linear, ReLU
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from src.utils import lower_tri, diag
# from src.neural_net.embedding import PositionalEncoding

class GaussianDensityNetwork(L.LightningModule):
    def __init__(self, d_x, d_theta, optimizer:torch.optim.Optimizer,
                 encoder:torch.nn.Module, mean_field):
        super().__init__()
        self.estimator = "gdn"
        # compute number of outputs
        self.dim = d_theta[0]
        self.optimizer = optimizer
        # assume diagonal covariance matrix
        if mean_field:
            d_output = self.dim * 2
        else:
            d_output = self.dim + self.dim*(self.dim + 1) // 2
        # eventually need to save this as an hparam if i am checkpointing models
        self.mean_field = mean_field
        self.val_losses = []
        # will this work with partial instantiation?
        if len(d_x) == 1: d_x = d_x[0]
        self.encoder = encoder(d_x, d_output)


    def forward(self, x):
        y = self.encoder(x)
        mu = y[:, :self.dim]
        sigma = y[:, self.dim:]
        # case one: unidimensional or mean field
        if self.dim == 1:
            sigma = torch.exp(sigma)
        elif self.mean_field:
            sigma = diag(torch.exp(sigma))
        else:
            sigma = lower_tri(sigma, self.dim)
            # force diagonal entries to be positive
            sigma.diagonal(dim1=-2, dim2=-1).copy_(
                sigma.diagonal(dim1=-2,dim2=-1).exp()
            )
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        x, theta = batch
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, theta = batch
        assert len(theta.shape) > 1
        mu, sigma = self(x)
        loss = self.gaussiannll(theta, mu, sigma)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        # why was this so difficult to figure out
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def gaussiannll(self, theta, mu, sigma):
        p = self.dim
        if p == 1:
            normal = Normal(mu, sigma)
            l = - normal.log_prob(theta)
        else:
            L = sigma
            mvn = MultivariateNormal(loc=mu, scale_tril=L)
            l = - mvn.log_prob(theta)

        return l.mean()
    

    def configure_optimizers(self):
        return self.optimizer(params = self.parameters())
    

    def predict_step(self, x):
        # this returns standard deviation
        mu, sigma = self(x)
        return mu, sigma


# class GaussianDensityNetwork(GaussianDensityNetworkBase):
#     def __init__(self, d_x, d_theta, d_model, optimizer,
#                  mean_field):
#         super().__init__(d_theta, optimizer, mean_field)

        
#         first_dim = d_x[0]

#         # TODO: do we need more flexibility in terms of layer widths?
#         # disadvantage is that the API becomes cumbersome if so

#         self.mean_field = mean_field
#         self.val_losses = []

        
#     def encoder(self, x):
#         return self.ff(x)
    
    
# class GaussianDensityTransformer(GaussianDensityNetworkBase):
#     def __init__(self, d_x, d_theta, d_model, lr, weight_decay,
#                 mean_field, n_heads, dropout, n_blocks):
#         super().__init__(d_theta, lr, weight_decay,
#                 mean_field)
        
#         self.pos_encode = PositionalEncoding(d_model)
#         encoder_layer = torch.nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dropout=dropout,
#             batch_first=True,
#             dim_feedforward=d_model*4 # it's a heuristic idk
#         )
#         norm = torch.nn.LayerNorm(d_model)
#         assert len(d_x) == 2
#         self.d_x = d_x
#         self.embed = Linear(d_x[0], d_model)
#         self.transformer = torch.nn.TransformerEncoder(encoder_layer, n_blocks, norm)
#         self.to_output = torch.nn.Sequential(
#             ReLU(),
#             Linear(d_model, d_output),
#         )
        
#     def encoder(self, x):
#         # transformer is expecting 
#         # (batch, seq, feature)
#         x = x.unflatten(-1, self.d_x).transpose(1, 2)
#         x = self.embed(x)
#         x = self.pos_encode(x)
#         y = self.transformer(x)
#         # pooling operation
#         # averaging makes the most intuitive sense imo
#         y = y.mean(dim=1)
#         return self.to_output(y)