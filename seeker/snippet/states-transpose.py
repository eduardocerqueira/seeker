#date: 2025-01-16T16:58:34Z
#url: https://api.github.com/gists/1a1aee9b17fa34b9c1a370d980f8b8c8
#owner: https://api.github.com/users/dessatel

#CML https://github.com/apple/coremltools/issues/2278
import torch
import torch.nn as nn
import coremltools as ct

class TestAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(2048, 2048, bias=False)
        self.wv = nn.Linear(2048, 2048, bias=False)
        self.wo = nn.Linear(128, 128, bias=False)

        cache_shape = (1, 16, 128, 128)

        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=torch.float32, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=torch.float32, device="cpu")
        )

    def forward(
        self, embedding, zt
    ):
        bsz, seqlen, _ = embedding.shape

        k, v = self.wk(embedding), self.wv(embedding)
        zt = zt[:, :, :seqlen, :] 
        
        if True:
            k = k.view(1, seqlen, 16, 128)
            v = v.view(1, seqlen, 16, 128)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # set to False for Error, True for work around
            work_around = True

            # add external tensor 
            # if transposed tensor is big you might need bigger op.
            # possibly tranpose/permute and slice_update are using the same op?
            # and it needs to complete execution prior to slice_update?
            if work_around: 
                zk = zt+k
                zv = zt+v
            else:
                zk = k
                zv = v
        else:
            # no transpose
            zk = k.view(1, 16, seqlen, 128)
            zv = v.view(1, 16, seqlen, 128)        

        self.v_cache[:, :, 0 : seqlen] = zv

        sum = self.k_cache+self.v_cache

        zt = torch.zeros(1, 16, 128, 128, dtype=zt.dtype)        
        return self.wo(sum) + zt

model_t = TestAttention().eval()


inputs = (
            torch.rand(1, 48, 16 * 128),
            torch.zeros(1, 16, 48, 128)
        )

traced_model = torch.jit.trace(model_t, inputs)

states = [ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, 16, 128, 128),
                    ),
                    name=v,
                ) for v in ['k_cache', 'v_cache']]
mlmodel = ct.convert(
    traced_model,
    inputs = [ct.TensorType(shape=(1, 48, 16 * 128)), ct.TensorType(shape=(1, 16, 128, 128))],
    outputs = [ ct.TensorType(name="op")],
    states=states,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
mlmodel.save("states-transpose.mlpackage")

mlmodel2 = ct.models.MLModel("states-transpose.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
state = mlmodel2.make_state()

# Run prediction
inputs = {
    "embedding": torch.rand(1, 48, 16 * 128).numpy(),
    "zt_1": torch.zeros(1, 16, 128, 128).numpy()
}
predictions = mlmodel2.predict(inputs, state)

# Print output shape and values
print("\nPrediction Results:")
print(f"Output shape: {predictions['op'].shape}")
print(f"Output values (first few):\n{predictions['op'].flatten()[:5]}")