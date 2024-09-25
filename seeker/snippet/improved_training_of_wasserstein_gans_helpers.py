#date: 2024-09-25T17:00:59Z
#url: https://api.github.com/gists/14d152a45eb4daf3f69ed872b6cbc9c4
#owner: https://api.github.com/users/MaximeVandegar

def sample_noise(batch_size, device):
    return torch.randn((batch_size, 128), device=device)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')