#date: 2024-12-17T17:06:12Z
#url: https://api.github.com/gists/4959429d3641d819df9d6cc016ce4c49
#owner: https://api.github.com/users/docsallover

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()