#date: 2022-07-26T17:04:01Z
#url: https://api.github.com/gists/9775fc2bf491e833c5bcd6663fe5cf38
#owner: https://api.github.com/users/ashhadulislam

PATH = './densenet169.pth'

# setup
model_ft = models.densenet169(pretrained=True,)
print(model_ft.classifier)
model_ft.classifier=nn.Linear(1664,len(classes))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)

torch.save(model_ft.state_dict(), PATH)

#load
model_ft3 = models.densenet169(pretrained=True)
model_ft3.classifier=nn.Linear(1664,len(classes))
model_ft3.to(device)

model_ft3.load_state_dict(torch.load(PATH,map_location=device))
model_ft3.eval()

#test
print(accuracy(model_ft3, test_loader))
