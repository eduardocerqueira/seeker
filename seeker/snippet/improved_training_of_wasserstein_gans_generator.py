#date: 2024-09-25T16:53:22Z
#url: https://api.github.com/gists/d50e9531c6e36c936e1913663afc95e2
#owner: https://api.github.com/users/MaximeVandegar

class UpResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super(UpResBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, 4 * output_dim, kernel_size=1, stride=1, bias=True),
            nn.PixelShuffle(2))
        self.network = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.dim = dim
        self.input_layer = nn.Linear(128, 4 * 4 * 8 * dim)
        self.model = nn.Sequential(UpResBlock(8 * dim, 8 * dim, 3),
                                   UpResBlock(8 * dim, 4 * dim, 3),
                                   UpResBlock(4 * dim, 2 * dim, 3),
                                   UpResBlock(2 * dim, 1 * dim, 3),
                                   nn.BatchNorm2d(1 * dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(1 * dim, 3, kernel_size=3, padding=1),
                                   nn.Tanh())

    def forward(self, noise):
        output = self.input_layer(noise)
        output = output.view(-1, 8 * self.dim, 4, 4)
        return self.model(output).view(-1, 3, 64, 64)