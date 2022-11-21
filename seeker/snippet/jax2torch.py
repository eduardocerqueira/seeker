#date: 2022-11-21T17:12:44Z
#url: https://api.github.com/gists/244a35e437b233e8888e54f588b0c046
#owner: https://api.github.com/users/seba-1511

import torch
import torch.utils.dlpack
import jax
import jax.dlpack

# A generic mechanism for turning a JAX function into a PyTorch function.

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch

def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax

def jax2torch(fun):

  class JaxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
      y_, ctx.fun_vjp = jax.vjp(fun, t2j(x))
      return j2t(y_)

    @staticmethod
    def backward(ctx, grad_y):
      grad_x_, = ctx.fun_vjp(t2j(grad_y))
      return j2t(grad_x_),

  return JaxFun.apply


# Here's a JAX function we want to interface with PyTorch code.

@jax.jit
def jax_square(x):
  return x ** 2

torch_square = jax2torch(jax_square)

# Let's run it on Torch data!
import numpy as np

x = torch.from_numpy(np.array([1., 2., 3.], dtype='float32'))
y = torch_square(x)
print(y)  # tensor([1., 4., 9.])

# And differentiate!
x = torch.tensor(np.array([1., 2., 3.], dtype='float32'), requires_grad=True)
y = torch.sum(torch_square(x))
y.backward()
print(x.grad)  # tensor([2., 4., 6.])