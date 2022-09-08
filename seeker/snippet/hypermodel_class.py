#date: 2022-09-08T17:23:17Z
#url: https://api.github.com/gists/c76878f78e8079aa038b89ad55ae96e7
#owner: https://api.github.com/users/Poulinakis-Konstantinos

class MyHyperModel(keras_tuner.HyperModel) :
    def build(self, hp, classes=37) : 
        model = keras.Sequential()
        model.add(layers.Input( (400,400,3)))
        model.add(layers.Resizing(128, 128, interpolation='bilinear'))
        # Whether to include normalization layer
        if hp.Boolean("normalize"):
            model.add(layers.Normalization())
        
        drop_rate = hp.Float("drop_rate", min_value=0.05, max_value=0.25, step=0.10)
        # Number of Conv Layers is up to tuning
        for i in range( hp.Int("num_conv", min_value=7, max_value=10, step=1)) :   
            # Tune hyperparams of each conv layer separately by using f"...{i}"
            model.add(layers.Conv2D(filters=hp.Int(f"filters_{i}", min_value=20, max_value=80, step=15),
                                    kernel_size= hp.Int(f"kernel_{i}", min_value=3, max_value=9, step=2),
                                    strides=1, padding='valid',
                                    activation=hp.Choice(f"conv_act_{i}", ["relu","leaky_relu", "sigmoid"] )))
            # Batch Norm and Dropout layers as hyperparameters to be searched
            if hp.Boolean("batch_norm"):
                model.add(layers.BatchNormalization())
            if hp.Boolean("dropout"):
                model.add(layers.Dropout(drop_rate))

        model.add(layers.Flatten())
        for i in range(2) #(hp.Int("num_dense", min_value=1, max_value=3, step=1)) :
            model.add(layers.Dense(units=hp.Choice("neurons", [150, 200]),
                                       activation=hp.Choice(['sigmoid', 'relu']))
            if hp.Boolean("batch_norm"):
                    model.add(layers.BatchNormalization())
            if hp.Boolean("dropout"):
                    model.add(layers.Dropout(drop_rate))
        
        # Last layer
        model.add(layers.Dense(classes, activation='softmax'))
        
        model.compile(optimizer=hp.Choice('optim',['adam','adamax']))
                      loss=hp.Choice("loss",["categorical_crossentropy","kl_divergence"]),
                      metrics = ['accuracy'])
        
        return model
    
    
    def fit(self, hp, model,x, *args, **kwargs) :
        
        return model.fit( x, 
                         *args,
                         shuffle=hp.Boolean("shuffle"),
                         **kwargs)
