#date: 2023-11-10T16:54:08Z
#url: https://api.github.com/gists/3821243ea24c05b940d71236625675e2
#owner: https://api.github.com/users/a1678991

# TensorRT
export TORCH_COMMAND="pip install -U pip wheel build setuptools pyproject.toml fonts ninja && \
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install -U --pre tensorrt --extra-index-url https://pypi.nvidia.com && \
pip install -U polygraphy onnx-graphsurgeon protobuf --extra-index-url https://pypi.ngc.nvidia.com && \
pip install -U 'httpx==0.24.*' protobuf==3.20.2"