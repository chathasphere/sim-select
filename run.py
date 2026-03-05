import hydra
from hydra.utils import instantiate
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from src.criticism import c2st, z_test

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(cfg):

    print("Instantiating simulator...")
    datamodule = instantiate(cfg.data, task=cfg.task, _convert_ = "partial")
    observed_data = datamodule.dataset.get_observed_data()
    
    print("Instantiating neural networks...")
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
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=cfg.callbacks.patience)]
    else:
        callbacks = None
    
    print("Instantiating trainer...")
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    
    if cfg.task == "estimation":
        posterior_params = model.predict_step(observed_data)
        # TODO: save results to yaml
        print(posterior_params)
    elif cfg.task == "criticism":
        x_val, mll_val = trainer.predict(model, datamodule.val_dataloader())[0]
        if cfg.c2st.check:
            # apply two sample classification test
            samples = model.sample(x_val.shape[0])
            if len(x_val.shape) > 2: x_val = x_val.flatten(1)
            score = c2st(x_val, samples)
            # oh whoops lol
            if (0.5 - cfg.c2st.tol) <= score <= (0.5 + cfg.c2st.tol):
                print(f"Marginal Density Estimator converged with score {score:.2f}")
            else:
                print(f"Score {score:.2f} is far from the desired near-chance performance.")
                print("Proceed with caution!")
        # default behavior should be: test for convergence
        # compute p-value a.k.a. marginal likelihood
        _, marginal_log_likelihood = model.predict_step(observed_data)
        p_value, reject_null = z_test(mll_val, marginal_log_likelihood)
        # TODO: save out results of criticism
        print(p_value)
        

if __name__ == "__main__":
    main()