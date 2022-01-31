#date: 2022-01-31T17:05:10Z
#url: https://api.github.com/gists/5b477dab42ed9039905275b143729b13
#owner: https://api.github.com/users/ndemir

def fit(model, epochs, train_loader, val_loader, loss_func, optimizer, lr_scheduler, val_metrics):
    loss_history = {'train': [], 'val': []}
    val_metrics_history = {k:[] for k in val_metrics}
    
    for epoch in tqdm(range(epochs)):
        model.train()
         
        loss_history_for_batch = []
        val_metrics_for_batch = {k:[] for k in val_metrics}
        
        for batch in train_loader:
            X, y = batch
            loss = loss_func(model(X.to(device)), y.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            loss_history_for_batch.append(t2np(loss))
            optimizer.step()
            
        lr_scheduler.step()

        loss_history['train'].append(np.mean(loss_history_for_batch))

        model.eval()
        with torch.no_grad():
            loss = evaluate(
                loader=val_loader, eval_f=loss_func, y_pred_f=model
            )
            loss_history['val'].append(loss)
            
            for metric in val_metrics:
                score = evaluate(
                    loader=val_loader, eval_f=val_metrics[metric], y_pred_f=Predict(model)
                )
                val_metrics_history[metric].append(score)
                
                

    return loss_history, val_metrics_history

def PredictProba(model):
    def f(x):
        return torch.softmax(model(x), 1)
    return f

def Predict(model):
    def f(x):
        probs = PredictProba(model)(x)
        return probs.argmax(dim=1)
    return f

def evaluate(loader, eval_f, y_pred_f):
    ret_list = []
    for batch in loader:
        X, y = batch
        ret = eval_f(y_pred_f(X.to(device)), y.to(device))
        ret_list.append(t2np(ret))
    
    return np.mean(ret_list)
        
def accuracy(y_pred, y_true):
    corrects = (y_true == y_pred)
    return corrects.sum()/corrects.shape[0]
    
t2np = lambda t: t.cpu().detach().numpy() if t.get_device()>=0 else t.detach().numpy()

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
val_metrics = {'accuracy': accuracy}


loss_history, val_metrics_history = fit(
    model=model, epochs=25, train_loader=train_loader, val_loader=val_loader,
    loss_func=loss, optimizer=optimizer, lr_scheduler=lr_scheduler,
    val_metrics=val_metrics
)

