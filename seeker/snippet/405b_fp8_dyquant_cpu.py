#date: 2024-07-16T17:06:17Z
#url: https://api.github.com/gists/600b22689fce917c6e8934f81ef55385
#owner: https://api.github.com/users/cli99

# https://github.com/neuralmagic/AutoFP8
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "databricks/llama-405b-instruct"
activation_scheme = "dynamic"
quantized_model_dir = pretrained_model_dir + "-FP82-" + activation_scheme

quantize_config = BaseQuantizeConfig(
    quant_method="fp8", activation_scheme=activation_scheme
)
model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantize_config=quantize_config,
    device_map="cpu",
)

model.quantize()
model.save_quantized(quantized_model_dir)
