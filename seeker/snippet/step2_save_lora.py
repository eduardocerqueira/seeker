#date: 2025-11-20T16:55:51Z
#url: https://api.github.com/gists/e4fab34eb8c13ef9b70a27de54fd1792
#owner: https://api.github.com/users/anubhaw2091

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch


def save_lora_adapter(
    base_model_id: str = None,  # If None, uses default below
    output_dir: str = "finetuned_lora",
    lora_r: int = 8,
    lora_alpha: int = 16
):
    """Save LoRA adapter after fine-tuning."""
    # Model configuration
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"  # Base model (general purpose) - DEFAULT
    INSTRUCT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Instruct version (better for instruction tasks)
    
    # Use provided model_id or default to BASE_MODEL
    if base_model_id is None:
        base_model_id = BASE_MODEL
        # base_model_id = INSTRUCT_MODEL  # Uncomment to use Instruct version
    print("=" * 60)
    print("Step 2: Saving LoRA Adapter")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Loading base model: {base_model_id}")
    
    # Load tokenizer
    tokenizer = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"p "**********"a "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        tokenizer.pad_token = "**********"
    
    # Load model
    # Use CPU for training (more stable, avoids meta device issues)
    USE_CPU_FOR_TRAINING = True  # Set to False to try MPS (may have issues)
    
    if USE_CPU_FOR_TRAINING:
        device = "cpu"
        dtype = torch.float32
        print(" Using CPU (recommended for stable fine-tuning on Mac)")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(" Using MPS (Apple Silicon GPU)")
        print(" Warning: MPS fine-tuning has known issues. CPU is more stable.")
    else:
        device = "cpu"
        dtype = torch.float32
        print("✓ Using CPU")
    
    # Load model with explicit device placement (avoid device_map="auto" to prevent meta device issues)
    print(f"\n Loading model with dtype={dtype} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=None,  # Explicit placement to avoid meta device
    )
    model = model.to(device)
    print(" Model loaded")
    
    # Configure and attach LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print(f"\n Saving LoRA adapter to: {output_path.absolute()}")
    
    # Ensure model is on CPU before saving (required to avoid meta device errors)
    print("   Moving model to CPU for saving...")
    model = model.to("cpu")
    
    # Save adapter and tokenizer
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print(" LoRA adapter saved")
    print(" Tokenizer saved")
    
    # Save metadata
    metadata = {
        "base_model": base_model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "output_path": str(output_path.absolute())
    }
    
    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(" Metadata saved")
    
    print("\n" + "=" * 60)
    print(" LoRA Adapter Saved Successfully!")
    print("=" * 60)
    print(f"\nAdapter location: {output_path.absolute()}")
    
    return str(output_path.absolute())


if __name__ == "__main__":
    save_lora_adapter()

