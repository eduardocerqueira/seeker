#date: 2021-12-29T17:07:58Z
#url: https://api.github.com/gists/9efb56cccd57f9c84910f02ccabf6fac
#owner: https://api.github.com/users/nicjac

class WandbCallback(Callback):
    "Saves model topology, losses & metrics"
    remove_on_fetch,order = True,Recorder.order+1
    # Record if watch has been called previously (even in another instance)
    _wandb_watch_called = False

    def __init__(self, log="gradients", log_preds=True, log_model=True, log_dataset=False, dataset_name=None, valid_dl=None, n_preds=36, seed=12345, reorder=True):
        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError('You must call wandb.init() before WandbCallback()')
        # W&B log step
        self._wandb_step = wandb.run.step - 1  # -1 except if the run has previously logged data (incremented at each batch)
        self._wandb_epoch = 0 if not(wandb.run.step) else math.ceil(wandb.run.summary['epoch']) # continue to next epoch
        store_attr('log,log_preds,log_model,log_dataset,dataset_name,valid_dl,n_preds,seed,reorder')

    def before_fit(self):
        "Call watch method to log model topology, gradients & weights"
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") and rank_distrib()==0
        if not self.run: return

        # Log config parameters
        log_config = self.learn.gather_args()
        _format_config(log_config)
        try:
            wandb.config.update(log_config, allow_val_change=True)
        except Exception as e:
            print(f'WandbCallback could not log config parameters -> {e}')

        if not WandbCallback._wandb_watch_called:
            WandbCallback._wandb_watch_called = True
            # Logs model topology and optionally gradients and weights
            wandb.watch(self.learn.model, log=self.log)

        # log dataset
        assert isinstance(self.log_dataset, (str, Path, bool)), 'log_dataset must be a path or a boolean'
        if self.log_dataset is True:
            if Path(self.dls.path) == Path('.'):
                print('WandbCallback could not retrieve the dataset path, please provide it explicitly to "log_dataset"')
                self.log_dataset = False
            else:
                self.log_dataset = self.dls.path
        if self.log_dataset:
            self.log_dataset = Path(self.log_dataset)
            assert self.log_dataset.is_dir(), f'log_dataset must be a valid directory: {self.log_dataset}'
            metadata = {'path relative to learner': os.path.relpath(self.log_dataset, self.learn.path)}
            log_dataset(path=self.log_dataset, name=self.dataset_name, metadata=metadata)

        # log model
        if self.log_model and not hasattr(self, 'save_model'):
            print('WandbCallback requires use of "SaveModelCallback" to log best model')
            self.log_model = False

        if self.log_preds:
            try:
                if not self.valid_dl:
                    #Initializes the batch watched
                    wandbRandom = random.Random(self.seed)  # For repeatability
                    self.n_preds = min(self.n_preds, len(self.dls.valid_ds))
                    idxs = wandbRandom.sample(range(len(self.dls.valid_ds)), self.n_preds)
                    if isinstance(self.dls,  TabularDataLoaders):
                        test_items = getattr(self.dls.valid_ds.items, 'iloc', self.dls.valid_ds.items)[idxs]
                        self.valid_dl = self.dls.test_dl(test_items, with_labels=True, process=False)
                    else:
                        test_items = [getattr(self.dls.valid_ds.items, 'iloc', self.dls.valid_ds.items)[i] for i in idxs]
                        self.valid_dl = self.dls.test_dl(test_items, with_labels=True)
                self.learn.add_cb(FetchPredsCallback(dl=self.valid_dl, with_input=True, with_decoded=True, reorder=self.reorder))
            except Exception as e:
                self.log_preds = False
                print(f'WandbCallback was not able to prepare a DataLoader for logging prediction samples -> {e}')

    def after_batch(self):
        "Log hyper-parameters and training loss"
        if self.training:
            self._wandb_step += 1
            self._wandb_epoch += 1/self.n_iter
            hypers = {f'{k}_{i}':v for i,h in enumerate(self.opt.hypers) for k,v in h.items()}
            wandb.log({'epoch': self._wandb_epoch, 'train_loss': to_detach(self.smooth_loss.clone()), 'raw_loss': to_detach(self.loss.clone()), **hypers}, step=self._wandb_step)

    def log_predictions(self, preds):
        inp,preds,targs,out = preds
        b = tuplify(inp) + tuplify(targs)
        x,y,its,outs = self.valid_dl.show_results(b, out, show=False, max_n=self.n_preds)
        wandb.log(wandb_process(x, y, its, outs), step=self._wandb_step)

    def after_epoch(self):
        "Log validation loss and custom metrics & log prediction samples"
        # Correct any epoch rounding error and overwrite value
        self._wandb_epoch = round(self._wandb_epoch)
        wandb.log({'epoch': self._wandb_epoch}, step=self._wandb_step)
        # Log sample predictions
        if self.log_preds:
            try:
                self.log_predictions(self.learn.fetch_preds.preds)
            except Exception as e:
                self.log_preds = False
                self.remove_cb(FetchPredsCallback)
                print(f'WandbCallback was not able to get prediction samples -> {e}')
        wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']}, step=self._wandb_step)         
            

    def after_fit(self):
        if self.log_model:
            if self.save_model.last_saved_path is None:
                print('WandbCallback could not retrieve a model to upload')
            else:
                log_model(self.save_model.last_saved_path, metadata=self.save_model.last_saved_metadata)
                
                for metadata_key in self.save_model.last_saved_metadata:
                    wandb.run.summary[f'best_{metadata_key}'] = self.save_model.last_saved_metadata[metadata_key]
                    
        self.run = True
        if self.log_preds: self.remove_cb(FetchPredsCallback)
        wandb.log({})  # ensure sync of last step
        self._wandb_step += 1