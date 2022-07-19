#date: 2022-07-19T17:12:18Z
#url: https://api.github.com/gists/7a6db264924c1562118f4c24014a4f68
#owner: https://api.github.com/users/ericd-1qbit

class UnetTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels: int = 3
        self.initial_conv_dim: int = 32
        self.initial_conv_kernel_size: int = 3
        self.padding: str = "same"
        self.initial_conv_layer = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.initial_conv_dim,
            kernel_size=self.initial_conv_kernel_size,
            padding=self.padding,
        )
        print(f"torch conv: {self.initial_conv_layer}")

    def forward(self, inputs):
        x = self.initial_conv_layer(inputs)
        return x


def get_dataset_image():
    from preprocessor.preprocess import Dataset

    TRAINING_FOLDER = "tests/test_data/images_for_test_dataset"
    dataset = Dataset(TRAINING_FOLDER, 128)
    print(dataset[0].size())  # [3, 128, 128]
    return dataset[0]


def _test_self_integrity(self) -> None:
    pass

def forward_pass_unet_torch(self, inputs: torch.Tensor = None) -> torch.Tensor:
    # batch N, channels C, depth D, height H, width W
    # torch style: N, C, H, W
    input = self.input_torch if inputs is None else inputs
    return self._unet_torch(input)

def forward_pass_unet_jax(self, inputs: jnp.ndarray) -> None:
    # batch N, channels C, depth D, height H, width W
    # jax style: N, H, W, C
    input = self.input_torch if inputs is None else inputs
    # apply(params, input_data) - This method performs forward pass through the jax network
    return self._unet_jax.apply(input)

to read these parameters, uncomment this:
jax.tree_map(jnp.shape, unfreeze(params)),


t_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding="valid")

kernel = t_conv.weight.detach().cpu().numpy()
bias = t_conv.bias.detach().cpu().numpy()

# [outC, inC, kH, kW] -> [kH, kW, inC, outC]
kernel = jnp.transpose(kernel, (2, 3, 1, 0))

key = random.PRNGKey(0)
x = random.normal(key, (1, 6, 6, 3))

variables = {"params": {"kernel": kernel, "bias": bias}}
j_conv = nn.Conv(features=4, kernel_size=(2, 2), padding="valid")
j_out = j_conv.apply(variables, x)

# [N, H, W, C] -> [N, C, H, W]
t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
t_out = t_conv(t_x)
# [N, C, H, W] -> [N, H, W, C]
t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))

np.testing.assert_almost_equal(j_out, t_out, decimal=6)

# float64 precision
from jax.config import config
config.update("jax_enable_x64", True)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)