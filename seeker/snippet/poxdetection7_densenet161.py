#date: 2022-07-26T16:57:19Z
#url: https://api.github.com/gists/5b7655a8bdb2d84d3a1fef15abe70006
#owner: https://api.github.com/users/ashhadulislam

PATH = './densenet161.pth'

# setup model
model_ft = models.densenet161(pretrained=True,)
model_ft.classifier=nn.Linear(2208,len(classes))
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train and save
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
torch.save(model_ft.state_dict(), PATH)

# laod model
model_ft3 = models.densenet161(pretrained=True)
model_ft3.classifier=nn.Linear(2208,len(classes))
model_ft3.to(device)
model_ft3.load_state_dict(torch.load(PATH,map_location=device))
model_ft3.eval()

# test model
print(accuracy(model_ft3, test_loader))
