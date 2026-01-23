import hydra
from hydra.utils import instantiate
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg):

    datamodule = instantiate(cfg.data, task=cfg.task, _convert_ = "partial")
    observed_data = datamodule.dataset.get_observed_data()
    
    if cfg.task == "estimation":
        # model the posterior p(theta | x)
        model = instantiate(cfg.model, d_x=datamodule.d_x, d_theta=datamodule.d_theta,
                            _convert_ = "all")
    elif cfg.task == "criticism":
        # model the marginal density p(x)
        model = instantiate(cfg.model, d_x = datamodule.d_x, _convert_ = "all")
        
    if cfg.log:
        wandb.init(reinit=False)
        logger = WandbLogger(project="sim-select")
    else:
        logger = None
        
    if cfg.callbacks.stop_early:
        callbacks = callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=cfg.callbacks.patience)]
    else:
        callbacks = None
    
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    
    if cfg.task == "estimation":
        posterior_params = model.predict_step(observed_data)
        # TODO: save results to yaml
        print(posterior_params)
    elif cfg.task == "criticism":
        # what is the output? marginal likelihood?
        pass

if __name__ == "__main__":
    main()