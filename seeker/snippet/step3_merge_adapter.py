#date: 2025-11-20T17:00:50Z
#url: https://api.github.com/gists/fea3ac820c875b2f5144776a59e451da
#owner: https://api.github.com/users/anubhaw2091

import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def merge_lora_adapter(
    base_model_id: str = None,  # If None, uses default below
    lora_adapter_path: str = "finetuned_lora",
    output_dir: str = "merged_model"
):
    """Merge LoRA adapter into full model."""
    # Model configuration
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"  # Base model (general purpose) - DEFAULT
    INSTRUCT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Instruct version (better for instruction tasks)
    
    # Use provided model_id or default to BASE_MODEL
    if base_model_id is None:
        base_model_id = BASE_MODEL
        # base_model_id = INSTRUCT_MODEL  # Uncomment to use Instruct version
    print("=" * 60)
    print("Step 3: Merging LoRA Adapter into Full Model")
    print("=" * 60)
    
    lora_path = Path(lora_adapter_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata if available
    metadata_path = lora_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "base_model" in metadata:
                base_model_id = metadata["base_model"]
        print(f" Loaded metadata from adapter")
    
    print(f"\n Loading base model: {base_model_id}")
    
    # Load base model
    # Use CPU for merging (more stable, avoids meta device issues)
    USE_CPU_FOR_TRAINING = True  # Set to False to try MPS (may have issues)
    
    if USE_CPU_FOR_TRAINING:
        device = "cpu"
        dtype = torch.float32
        print(" Using CPU (recommended for stable operations on Mac)")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(" Using MPS (Apple Silicon GPU)")
        print(" Warning: MPS may have issues with merging. CPU is more stable.")
    else:
        device = "cpu"
        dtype = torch.float32
        print(" Using CPU")
    
    # Load model with explicit device placement (avoid device_map="auto" to prevent meta device issues)
    print(f"\n Loading base model with dtype={dtype} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=None,  # Explicit placement to avoid meta device
    )
    base_model = base_model.to(device)
    print(" Base model loaded")
    
    # Load tokenizer
    tokenizer = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"p "**********"a "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        tokenizer.pad_token = "**********"
    print(" Tokenizer loaded")
    
    # Load and merge LoRA adapter
    print(f"\n Loading LoRA adapter from: {lora_path.absolute()}")
    
    # Ensure model is on CPU before loading adapter (required for PEFT)
    if device != "cpu":
        print("   Moving model to CPU for adapter loading...")
        base_model = base_model.to("cpu")
        device = "cpu"
    
    try:
        merged_model = PeftModel.from_pretrained(base_model, str(lora_path))
        print(" LoRA adapter loaded")
    except KeyError as e:
        print(f"\n Error loading adapter: {e}")
        print("   This may indicate a model structure mismatch.")
        print("   Ensure the base model matches the one used for fine-tuning.")
        raise
    
    print("\n Merging adapter into base model...")
    merged_model = merged_model.merge_and_unload()
    print(" Adapter merged (LoRA deltas baked into weights)")
    
    # Save merged model
    print(f"\n Saving merged model to: {output_path.absolute()}")
    
    # Ensure model is on CPU before saving
    if device != "cpu":
        print("   Moving model to CPU for saving...")
        merged_model = merged_model.to("cpu")
    
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(" Merged model saved")
    print(" Tokenizer saved")
    
    # Save conversion metadata
    conversion_metadata = {
        "base_model": base_model_id,
        "lora_adapter_path": str(lora_path.absolute()),
        "merged_model_path": str(output_path.absolute()),
        "format": "huggingface",
        "ready_for_conversion": True,
        "next_step": "Convert to GGUF using llama.cpp"
    }
    
    with open(output_path / "conversion_metadata.json", "w") as f:
        json.dump(conversion_metadata, f, indent=2)
    print(" Conversion metadata saved")
    
    print("\n" + "=" * 60)
    print(" Model Merged Successfully!")
    print("=" * 60)
    print(f"\nMerged model location: {output_path.absolute()}")
    print("\nNext step: Convert to GGUF using step4_convert_to_gguf.sh")
    print("  Default (uses 'merged_model', 'mymodel', and 'f16'):")
    print("    bash step4_convert_to_gguf.sh")
    print("  With custom parameters:")
    print(f"    bash step4_convert_to_gguf.sh <merged_model_path> <model_name> <quant_type>")
    print("  Valid quant types: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto")
    print("  Example:")
    print(f"    bash step4_convert_to_gguf.sh {output_path.name} mymodel f16")
    print("  Note: GGUF will be saved to llama.cpp/custom_models/<model_name>/<model_name>.gguf")
    print("  Note: For q4_k_m, convert to f16 first, then use llama-quantize for further quantization")
    
    return str(output_path.absolute())


if __name__ == "__main__":
    merge_lora_adapter()

