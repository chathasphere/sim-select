import hydra
from hydra.utils import instantiate
from src.utils import DataModule
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg):

    dataset = instantiate(cfg.simulator, _convert_ = "all")

    if cfg.train.batch_size is None:
        batch_size = cfg.simulator.n_sample
    else:
        batch_size = cfg.train.batch_size
    datamodule = DataModule(
        dataset, cfg.train.seed, batch_size, cfg.train.train_frac
        )
    
    model = instantiate(cfg.model, d_x=dataset.d_x, d_theta=dataset.d_theta,
                        _convert_ = "all")
    
    
    if cfg.log:
        wandb.init(reinit=False)
        logger = WandbLogger(project="sim-select")
    else:
        logger = None
        
    if cfg.train.stop_early:
        callbacks = callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=cfg.train.patience)]
    else:
        callbacks = None
    trainer = L.Trainer(max_epochs=cfg.train.max_epochs, logger=logger,
                        devices=cfg.train.devices,
                        log_every_n_steps=cfg.train.log_freq, callbacks=callbacks,
                        fast_dev_run=cfg.fast_dev_run)
    

    trainer.fit(model, datamodule=datamodule)
    

if __name__ == "__main__":
    main()