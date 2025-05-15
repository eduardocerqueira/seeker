#date: 2025-05-15T17:07:05Z
#url: https://api.github.com/gists/06ae4221e1927cdb3d84c5b7fb7b9007
#owner: https://api.github.com/users/mgagvani

import onnx
import onnx.helper as helper
import sys

onnx_path = 'model_proc.static_int8.onnx'

onnx_model = onnx.load(onnx_path)

# Create a new list for opset imports
new_opset_imports = []
for imp in onnx_model.opset_import:
    if imp.domain != "ai.onnx.ml":
        new_opset_imports.append(imp)

# Add the desired ai.onnx.ml opset
new_opset_imports.append(helper.make_opsetid("ai.onnx.ml", 3))

# Create a new model proto, assign the graph from the old model,
# and then set the new opset imports.
# This is a way to work around limitations with modifying the opset_import field directly.
new_model = helper.make_model(
    onnx_model.graph,
    producer_name=onnx_model.producer_name,
    producer_version=onnx_model.producer_version,
    opset_imports=new_opset_imports, # Use the new list here
    model_version=onnx_model.model_version,
    doc_string=onnx_model.doc_string,
    # Add other fields from onnx_model if they are important for your model
    # e.g., metadata_props=onnx_model.metadata_props
)
# If your model has ir_version, domain, etc., set them as well:
if hasattr(onnx_model, 'ir_version'):
    new_model.ir_version = onnx_model.ir_version
if hasattr(onnx_model, 'domain'):
    new_model.domain = onnx_model.domain
if hasattr(onnx_model, 'metadata_props'):
    for prop in onnx_model.metadata_props:
        new_model.metadata_props.add().CopyFrom(prop)


# sanity‐check & overwrite file
onnx.checker.check_model(new_model)
onnx.save(new_model, onnx_path)
print(f"Patched ai.onnx.ml → opset 3 in {onnx_path}")

for op_imp in new_model.opset_import:
   print(f"  Domain: <{op_imp.domain if op_imp.domain else 'ai.onnx'}>, Version: {op_imp.version}")