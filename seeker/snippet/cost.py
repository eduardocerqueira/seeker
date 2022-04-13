#date: 2022-04-13T16:45:15Z
#url: https://api.github.com/gists/5e48c1bdad095cc99bfbc83706daa152
#owner: https://api.github.com/users/jmduarte

from finn.util.inference_cost import inference_cost

models = ["models/ImageNet/Brevitas_FINN_mobilenet/mobilenet_4W4A.onnx",
          "models/CIFAR10/Brevitas_FINN_CNV/CNV_1W1A.onnx",
          "models/CIFAR10/Brevitas_FINN_CNV/CNV_1W2A.onnx",
          "models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx",
          "models/MNIST/Brevitas_FINN_TFC/TFC/TFC_1W1A.onnx",
          "models/MNIST/Brevitas_FINN_TFC/TFC/TFC_1W2A.onnx",
          "models/MNIST/Brevitas_FINN_TFC/TFC/TFC_2W2A.onnx"]

final_onnx_path = "tmp.onnx"
cost_dict_path = "tmp_model_cost.json"
for export_onnx_path in models:
    try:
        inference_cost(
            export_onnx_path,
            output_json=cost_dict_path,
            output_onnx=final_onnx_path,
            preprocess=True,
            discount_sparsity=False,
        )
    except Exception as e:
        print(e)