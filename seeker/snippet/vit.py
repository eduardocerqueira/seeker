#date: 2023-11-22T16:43:09Z
#url: https://api.github.com/gists/93d2e1bbaf4d10f42c5be264948a0d77
#owner: https://api.github.com/users/Camaltra

class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.embeding_layer = Embeding(
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            batch_size=cfg.batch_size,
            dropout_p=cfg.dropout_embedding,
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embed_dim=cfg.embed_dim,
                    num_of_head=cfg.num_of_head,
                    mlp_hidden_size=cfg.mlp_hidden_size,
                    dropout_p=cfg.dropout_linear,
                )
                for _ in range(cfg.num_layer_transformer)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=cfg.embed_dim),
            nn.Linear(in_features=cfg.embed_dim, out_features=cfg.num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded_imgs = self.embeding_layer(x)
        return self.classifier(self.transformer_encoder(embedded_imgs)[:, 0])