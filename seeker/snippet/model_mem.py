#date: 2025-01-01T16:29:34Z
#url: https://api.github.com/gists/4693e96d2739c947e0c807fb6b8ad847
#owner: https://api.github.com/users/Sinjhin

'''
M = ((P * 4B) / (32 / Q)) * 1.2

Where:

- M = Mem needed in GB
- P = num of params
- 4B = 4 bytes per parameter (@32 quant)
- 32 = bits in 4 bytes
- Q = model quantization
- 1.2 = 20% overhead for other stuff needed in mem

Q is:

- FP32 (32-bit floating point): 4 bytes per parameter
- FP16 (half/BF16) (16-bit floating point): 2 bytes per parameter
- INT8 (8-bit integer): 1 byte per parameter
- INT4 (4-bit integer): 0.5 bytes per parameter

e.x. Llama 70B loaded at 16 bits would be:

- ((70 * 4bytes) / (32 / 16)) * 1.2 = 280 / 2 * 1.2 = 168 GB
'''

quant = 8
param = 70
mem = 128

def getMem(p, q):
    return ((p * 4) / (32 / q)) * 1.2
    
def getParam(m, q):
    return ((m / 1.2) * (32 / q)) / 4
    
mem_needed = getMem(param, quant)
print(f"Mem needed for {param}B parameter model")
print(f"at {quant}bit quantization is: {(mem_needed):.2f}GB")
print("")
param_max = getParam(mem, quant)
print(f"With {mem}GB of memory")
print(f"at {quant}bit quantization")
print(f"you can run up to a {(param_max):.2f}B param model")
