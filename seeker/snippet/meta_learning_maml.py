#date: 2021-09-28T16:58:54Z
#url: https://api.github.com/gists/93fc0b09edc3e354975e3e028fdf3e92
#owner: https://api.github.com/users/tchaton

  adaptation_loss = model_clone(task_shots)

  # more logic might be required to ensure traceability
  # of the gradients.
  adaptation_grads = torch.autograd.grad(
    adaptation_loss, retain_graph=True, create_graph=True)

  # use functional optimizer API from `PyTorch`
  F.sgd(model_clone.parameters(), adaptation_grads)

  meta_loss += model_clone(adaptation_loss)