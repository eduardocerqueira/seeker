#date: 2022-06-16T16:54:53Z
#url: https://api.github.com/gists/40123a1e3e37e90cbbbe2aee19d479f7
#owner: https://api.github.com/users/krsnewwave

import gc

def objective(trial):
    ### Dataset section
    # see https://gist.github.com/krsnewwave/273b9cafa4813771791f076cee32c2e4#file-nvtabular_movielens_main_loop_functions-py-L2
    train_loader, valid_loader = create_loaders(train_dataset, valid_dataset)
    ### Model section
    # see https://gist.github.com/krsnewwave/273b9cafa4813771791f076cee32c2e4#file-nvtabular_movielens_main_loop_functions-py-L29
    epochs = 1
    patience = 3
    model = create_model(trial, epochs, patience)
    
    ### trainer specific
    # see https://gist.github.com/krsnewwave/273b9cafa4813771791f076cee32c2e4#file-nvtabular_movielens_main_loop_functions-py-L69
    trainer = create_trainer(epochs, patience)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders = valid_loader)
    trainer.logger.experiment.end()
    
    ## validation metrics
    val_metrics = trainer.test(dataloaders=valid_loader)
    
    # cleanup
    del model
    gc.collect()
    return val_metrics[0]["test_precision"]
    
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=6)