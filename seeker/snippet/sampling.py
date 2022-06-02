#date: 2022-06-02T16:58:04Z
#url: https://api.github.com/gists/ed64b2f256c3d2fd382b6d32515608eb
#owner: https://api.github.com/users/vivekkhandelwal1

import torch
from tqdm.auto import trange

from . import utils


# DDPM/DDIM sampling

def sample(model, x, steps, eta, extra_args, callback=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i]).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


def cond_sample(model, x, steps, eta, extra_args, cond_fn, callback=None):
    """Draws guided samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps), disable=None):

        # Get the model output
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                v = model(x, ts * steps[i])

            pred = x * alphas[i] - v * sigmas[i]

            # Call the callback
            if callback is not None:
                callback({'x': x, 'i': i, 't': steps[i], 'v': v.detach(), 'pred': pred.detach()})

            if steps[i] < 1:
                cond_grad = cond_fn(x, ts * steps[i], pred).detach()
                v = v.detach() - cond_grad * (sigmas[i] / alphas[i])
            else:
                v = v.detach()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < len(steps) - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


def reverse_sample(model, x, steps, extra_args, callback=None):
    """Finds a starting latent that would produce the given image with DDIM
    (eta=0) sampling."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    alphas, sigmas = utils.t_to_alpha_sigma(steps)

    # The sampling loop
    for i in trange(len(steps) - 1, disable=None):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * steps[i]).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # Call the callback
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'v': v, 'pred': pred})

        # Recombine the predicted noise and predicted denoised image in the
        # correct proportions for the next step
        x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    return x


# PNDM sampling (see https://openreview.net/pdf?id=PlKWVd2yBkY)


def eps_model_fn(model, x, t):
    alphas, sigmas = utils.t_to_alpha_sigma(t)
    v = model(x, t)
    eps = x * sigmas[:, None, None, None] + v * alphas[:, None, None, None]
    return eps

def make_autocast_model_fn(model, enabled=True):
    def autocast_model_fn(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled):
            return model(*args, **kwargs).float()
    return autocast_model_fn


def transfer(x, eps, t_1, t_2):
    alphas, sigmas = utils.t_to_alpha_sigma(t_1)
    next_alphas, next_sigmas = utils.t_to_alpha_sigma(t_2)
    pred = (x - eps * sigmas[:, None, None, None]) / alphas[:, None, None, None]
    x = pred * next_alphas[:, None, None, None] + eps * next_sigmas[:, None, None, None]
    return x, pred


def prk_step(model, x, t_1, t_2):
    t_mid = (t_2 + t_1) / 2
    eps_1 = eps_model_fn(model, x, t_1)
    x_1, _ = transfer(x, eps_1, t_1, t_mid)
    eps_2 = eps_model_fn(model, x_1, t_mid)
    x_2, _ = transfer(x, eps_2, t_1, t_mid)
    eps_3 = eps_model_fn(model, x_2, t_mid)
    x_3, _ = transfer(x, eps_3, t_1, t_2)
    eps_4 = eps_model_fn(model, x_3, t_2)
    eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms_step(model, x, old_eps, t_1, t_2):
    eps = eps_model_fn(model, x, t_1)
    eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


def prk_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth-order
    Pseudo Runge-Kutta."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    for i in range(len(steps) - 1):
        x, _, pred = prk_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def plms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    # model = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in range(len(steps) - 1):
        if len(old_eps) < 3:
            with torch.cuda.amp.autocast():
                x, eps, pred = prk_step(model, x, steps[i] * ts, steps[i + 1] * ts)
        else:
            with torch.cuda.amp.autocast():
                x, eps, pred = plms_step(model, x, old_eps, steps[i] * ts, steps[i + 1] * ts)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def pie_step(model, x, t_1, t_2, extra_args):
    eps_1 = eps_model_fn(model, x, t_1)
    x_1, _ = transfer(x, eps_1, t_1, t_2)
    eps_2 = eps_model_fn(model, x_1, t_2)
    eps_prime = (eps_1 + eps_2) / 2
    x_new, pred = transfer(x, eps_prime, t_1, t_2)
    return x_new, eps_prime, pred


def plms2_step(model, x, old_eps, t_1, t_2, extra_args):
    eps = eps_model_fn(model, x, t_1)
    eps_prime = (3 * eps - old_eps[-1]) / 2
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)
    return x_new, eps, pred


def pie_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using second-order
    Pseudo Improved Euler."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    for i in range(len(steps) - 1):
        x, _, pred = pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def plms2_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using second order
    Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in range(len(steps) - 1):
        if len(old_eps) < 1:
            x, eps, pred = pie_step(model_fn, x, steps[i] * ts, steps[i + 1] * ts, extra_args)
        else:
            x, eps, pred = plms2_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x
