#date: 2022-06-16T16:53:56Z
#url: https://api.github.com/gists/273b9cafa4813771791f076cee32c2e4
#owner: https://api.github.com/users/krsnewwave

# (1) Create loaders
def create_loaders(train_dataset, valid_dataset):
     # dataset and loaders
    train_iter = TorchAsyncItr(
        train_dataset,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        conts=NUMERIC_COLUMNS,
        labels=["rating"],
    )

    train_loader = DLDataLoader(
        train_iter, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
    )

    valid_iter = TorchAsyncItr(
        valid_dataset,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        conts=NUMERIC_COLUMNS,
        labels=["rating"],
    )
    valid_loader = DLDataLoader(
        valid_iter, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
    )
    return train_loader, valid_loader

# (2) Create models and hyperparameter search space
def create_model(trial, epochs, patience):
    # embeddings shape
    embedding_size = int(trial.suggest_discrete_uniform('embedding_size', 128, 512, 64))
    mh_embedding_size = 16
    embedding_table_shape = create_embeddings_shape(embedding_size, mh_embedding_size, user_id_size, item_id_size, genre_size)
    
    # embeddings dropout
    emb_dropout = trial.suggest_float("emb_dropout", 0.0, 0.6, step=0.1)
    
    # hidden dims
    hidden_dims_shape = trial.suggest_int("hidden_dims_shape", 128, 512, step=32, log=False)
    layer_hidden_dims = [hidden_dims_shape, hidden_dims_shape]
    
    # dropout (dependent on hidden dims)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.1)
    layer_dropout_rates = [dropout_rate, dropout_rate]
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    
    hyperparams = {
        "num_epochs": epochs,
        "patience": patience,
        "embedding_table_shape" : embedding_table_shape,
        "learning_rate": learning_rate,
        "wd" : wd,
        "layer_hidden_dims" : layer_hidden_dims,
        "layer_dropout_rates" : layer_dropout_rates,
        "emb_dropout" : emb_dropout
    }
    
    model = WideAndDeepMultihot(hyperparams,
                                CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
                                NUMERIC_COLUMNS, 
                                "rating",
                                num_continuous=0,
                                batch_size = BATCH_SIZE)
    return model

# create trainer
def create_trainer(epochs, patience):
    comet_logger = CometLogger(
        api_key=API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT_NAME,
        display_summary_level=0
    )
    callbacks = [pl.callbacks.EarlyStopping("val_precision", mode='max', patience=patience)]
    trainer = pl.Trainer(accelerator="auto", devices=1, callbacks=callbacks, enable_progress_bar=False,
                         max_epochs=epochs, log_every_n_steps=100, logger=comet_logger)
    return trainer