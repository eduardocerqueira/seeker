#date: 2024-09-25T16:58:50Z
#url: https://api.github.com/gists/96c3c1d5e235e76d0379f7065e602068
#owner: https://api.github.com/users/MaximeVandegar

def train(generator, critic, generator_optimizer, critic_optimizer, dataloader, nb_epochs, ncritic=5, lambda_gp=10.):
    training_loss = {'generative': [], 'critic': []}
    dataset_iter = iter(dataloader)

    for epoch in tqdm(range(nb_epochs)):
        k = (20 * ncritic) if ((epoch < 25) or (epoch % 500 == 0)) else ncritic
        for _ in range(k):

            # Sample a batch from the real data
            try:
                x = next(dataset_iter).to(device)
            except:
                dataset_iter = iter(dataloader)
                x = next(dataset_iter).to(device)
            batch_size = x.shape[0]
            # Sample a batch of prior samples
            z = sample_noise(batch_size, device)

            critic_optimizer.zero_grad()
            x_tilde = generator(z).detach()
            eps = torch.rand((x_tilde.shape[0], 1, 1, 1), device=device)
            x_hat = Variable(eps * x + (1 - eps) * x_tilde, requires_grad=True)
            loss = -(critic(x).squeeze(0) - critic(x_tilde).squeeze(0)).mean()

            gradients = torch.autograd.grad(outputs=critic(x_hat).squeeze(0), inputs=x_hat, grad_outputs=torch.ones(
                (x_hat.shape[0]), device=x_hat.device), create_graph=True, retain_graph=True)[0]
            gradient_penalty = ((gradients.reshape(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            loss = loss + lambda_gp * gradient_penalty
            loss.backward()
            critic_optimizer.step()

            training_loss['critic'].append(loss.item())

        # Train the generator

        # Sample a batch of prior samples
        z = sample_noise(batch_size, device)

        # Update the generator by descending its stochastic gradient
        loss = -critic(generator(z)).mean(0)

        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(loss.item())
    return training_loss