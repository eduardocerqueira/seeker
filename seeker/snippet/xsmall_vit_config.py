#date: 2023-11-22T16:53:42Z
#url: https://api.github.com/gists/bb1530831c740419465faf36e457ae2e
#owner: https://api.github.com/users/Camaltra

XSMALLVITCONFIG = ViTConfig(
    num_layer_transformer=2,
    embed_dim=768,
    mlp_hidden_size=768,
    dropout_linear=0.1,
    dropout_embedding=0.1,
    num_of_head=12,
    patch_size=16,
    batch_size=128,
    image_size=(256, 256),
    num_class=1,
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    learning_rate=1e-4,
    num_epoch=10,
)