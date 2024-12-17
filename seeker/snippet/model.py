#date: 2024-12-17T16:52:32Z
#url: https://api.github.com/gists/a993f5f82670d3a9b82946f288d16b16
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