#date: 2025-07-15T17:05:28Z
#url: https://api.github.com/gists/fc899dd9ff7fe357065c35cb0120d145
#owner: https://api.github.com/users/eustlb

from datasets import load_dataset, Audio
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
import os
import torch
from whisper.normalizers import EnglishTextNormalizer
import jiwer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch_device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"

BATCH_SIZE = 32
MODEL_ID = "/scratch/voxtral-mini-converted"

test_set = load_dataset("hf-audio/esb-datasets-test-only-sorted", "librispeech", split="test.clean")
test_set = test_set.cast_column("audio", Audio(sampling_rate=16000))

processor = VoxtralProcessor.from_pretrained(MODEL_ID)
model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, device_map=torch_device, torch_dtype=torch.bfloat16)

def eval_batch(batch):    
    inputs = processor.apply_transcrition_request(
        language="en", audio=[el['array'] for el in batch["audio"]], format=[el.metadata.codec.upper() for el in batch["audio"]]
    )
    inputs.to(torch_device, dtype=torch.bfloat16)
    outputs = "**********"=10000)
    decoded_outputs = processor.batch_decode(outputs[: "**********":], skip_special_tokens=True)

    return {
        "references": batch["text"],
        "predictions": decoded_outputs,
    }

infered_test_set = test_set.map(eval_batch, batched=True, batch_size=BATCH_SIZE, remove_columns=test_set.column_names)
infered_test_set.save_to_disk("infered_test_set")

normalizer = EnglishTextNormalizer()

normalized_refs = [normalizer(ref) for ref in infered_test_set["references"]]
normalized_hyps = [normalizer(hyp) for hyp in infered_test_set["predictions"]]

sum_wer = sum(jiwer.wer(ref, hyp) for ref, hyp in zip(normalized_refs, normalized_hyps))

print(f"mean WER: {sum_wer / len(infered_test_set)}")
print(f"Courpus WER: {jiwer.wer(normalized_refs, normalized_hyps)}")
wer(normalized_refs, normalized_hyps)}")
