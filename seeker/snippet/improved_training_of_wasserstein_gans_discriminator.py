#date: 2024-09-25T16:54:19Z
#url: https://api.github.com/gists/d976eb1003d96f0e72cde7623429f3c0
#owner: https://api.github.com/users/MaximeVandegar

class DownResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super(DownResBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, bias=True),
            nn.AvgPool2d(2))
        self.network = nn.Sequential(
            nn.InstanceNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.InstanceNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=filter_size, padding=filter_size // 2),
            nn.AvgPool2d(2))

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        output = self.network(inputs)
        return shortcut + output
        
        
class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.dim = dim

        self.input_conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.model = nn.Sequential(DownResBlock(dim, 2 * dim, 3),
                                   DownResBlock(2 * dim, 4 * dim, 3),
                                   DownResBlock(4 * dim, 8 * dim, 3),
                                   DownResBlock(8 * dim, 8 * dim, 3),)
        self.output_linear = nn.Linear(4 * 4 * 8 * dim, 1)

    def forward(self, inputs):
        output = self.input_conv(inputs)
        output = self.model(output)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        return self.output_linear(output).view(-1)