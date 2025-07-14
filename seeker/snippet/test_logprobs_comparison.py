#date: 2025-07-14T16:59:01Z
#url: https://api.github.com/gists/9c9446520cc742ca8347207842ba9fe5
#owner: https://api.github.com/users/LucasWilkinson

import unittest
import torch
import gc
import numpy as np
import transformers
import vllm
import parameterized
from dataclasses import dataclass
from typing import Dict, List, Union, Generic, List, Optional, TypeVar
from transformers import PreTrainedModel


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


T = TypeVar("T")
# torch.set_printoptions(precision=2, sci_mode=False)


@dataclass
class PackedSequences(Generic[T]):
    query_responses: np.ndarray
    """packed query and response (batch_size, pack_length)"""
    attention_masks: np.ndarray
    """3D attention mask for packed sequences (batch_size, pack_length, pack_length);
    it basically uses a intra-document mask for each query response pair;
    see https://huggingface.co/blog/sirluk/llm-sequence-packing for more details
    """
    response_masks: np.ndarray
    """response mask for packed sequences (batch_size, pack_length)"""
    original_responses: np.ndarray
    """need the original response for broadcast (batch_size, response_length)"""
    tool_masks: Optional[np.ndarray] = None
    """tool mask for packed sequences (batch_size, pack_length)"""
    advantages: Optional[np.ndarray] = None
    """packed advantages (batch_size, pack_length) (to be filled in by the main process)"""
    num_actions: Optional[np.ndarray] = None
    """packed number of actions (batch_size, pack_length)"""
    position_ids: Optional[np.ndarray] = None
    """packed position ids (batch_size, pack_length)"""
    packed_seq_lens: Optional[np.ndarray] = None
    """packed sequence lengths (batch_size, pack_length)"""
    dones: Optional[np.ndarray] = None
    """packed dones (batch_size, pack_length), specifies the sequence boundaries
    E.g., [0, 0, 0, 0, 1, 0, 0, 0, 0, 2] means the first sequence ends at index 4, and the 
    second sequence ends at index 9
    """
    rewards: Optional[np.ndarray] = None
    """packed rewards (batch_size, pack_length)"""


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def pack_sequences(
    queries: "**********": List[List[int]], masks: List[List[int]], pack_length: int, pad_token_id: int
) -> PackedSequences:
    assert not any(pad_token_id in query for query in queries)
    # TODO: "**********"
    # assert not any(pad_token_id in response for response in responses)

    query_responses = []
    tool_masks = []
    attention_masks = []
    response_masks = []
    dones = []
    num_actions = []
    packed_seq_lens = []
    cur_data = []
    cur_tool_mask = []
    cur_response_mask = []
    cur_num_actions = []
    cur_packed_seq_lens = []
    cur_attention_mask = []
    cur_dones = []
    offset = 0
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        mask = masks[i]
        # remove padding (but using vllm so this should not be needed, but just in case)
        query_tool_mask = "**********"= pad_token_id]
        query = "**********"= pad_token_id]
        response_tool_mask = "**********"= pad_token_id]
        response = "**********"= pad_token_id]
        query_response = query + response
        mask = query_tool_mask + response_tool_mask
        if len(query_response) + len(cur_data) > pack_length:
            query_responses.append(cur_data)
            tool_masks.append(cur_tool_mask)
            response_masks.append(cur_response_mask)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            dones.append(cur_dones)
            cur_data = []
            cur_tool_mask = []
            cur_response_mask = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            cur_dones = []
            offset = i
        cur_data.extend(query_response)
        cur_tool_mask.extend(mask)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))

        # @vwxyzjn: "**********"
        # the actual number should corresponds to the response's index.
        cur_response_mask.extend([0 for _ in range(len(query))] + [i + 1 for _ in range(len(response))])
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])
        cur_dones.extend([0 for _ in range(len(query) + len(response) - 1)] + [i + 1])

    # Handle leftover data
    if len(cur_data) > 0:
        query_responses.append(cur_data)
        tool_masks.append(cur_tool_mask)
        response_masks.append(cur_response_mask)
        attention_masks.append(cur_attention_mask)
        num_actions.append(cur_num_actions)
        packed_seq_lens.append(cur_packed_seq_lens)
        dones.append(cur_dones)
    attention_masks_list = [torch.tensor(t) for t in attention_masks]
    return PackedSequences(
        query_responses=[torch.tensor(t) for t in query_responses],
        attention_masks=attention_masks_list,
        position_ids=[reset_position_ids(t.unsqueeze(0)).squeeze(0) for t in attention_masks_list],
        response_masks=[torch.tensor(t) for t in response_masks],
        original_responses=responses,
        num_actions=[torch.tensor(t) for t in num_actions],
        packed_seq_lens=[torch.tensor(t) for t in packed_seq_lens],
        dones=[torch.tensor(t) for t in dones],
        tool_masks=[torch.tensor(t) for t in tool_masks],
    )


MAX_TOKENS = "**********"
SEED = 42
PACK_LENGTH = 64
DTYPE = "bfloat16"


class TestLogprobsComparison(unittest.TestCase):
    """Test logprobs calculation and comparison between HuggingFace and vLLM."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hamishivi/qwen3_openthoughts2", "The capital of France is"),
        ("hamishivi/qwen3_openthoughts2", "The weather today is"),
        ("hamishivi/qwen3_openthoughts2", "Machine learning is"),
    ])
    def test_vllm_hf_logprobs_match_large(self, model_name, prompt):
        """Test that vLLM and HuggingFace produce matching logprobs for large models (GPU only)."""

        tokenizer = "**********"
        query = "**********"
        
        # Get vLLM logprobs
        vllm_output = _get_vllm_logprobs(model_name, query)
        gc.collect()
        torch.cuda.empty_cache()
        packed_sequences = pack_sequences(
            queries=[query],
            responses=[vllm_output["response"]],
            # This mask is the tool use mask, which we mock to be all ones, as done in grpo_fast.py
            # when tooluse is disabled.
            masks=[[1] * len(vllm_output["response"])],
            pack_length=PACK_LENGTH,
            pad_token_id= "**********"
        )

        hf_logprobs = _get_hf_logprobs(model_name, query,
                                       vllm_output["response"],
                                       packed_sequences.query_responses[0],
                                       packed_sequences.attention_masks[0],
                                       packed_sequences.position_ids[0],
                                       tokenizer.pad_token_id)
        vllm_logprobs = vllm_output["logprobs"]
        
        # Check that the tokens being scored match
        packed_response_tokens = packed_sequences.query_responses[0][len(query): "**********"
        
        self.assertEqual(len(vllm_logprobs), len(vllm_output["response"]))
        self.assertEqual(len(vllm_logprobs), len(hf_logprobs), f'{vllm_logprobs=}\n{hf_logprobs=}')
        
        # Verify tokens match before comparing logprobs
        self.assertEqual(vllm_output['response'], packed_response_tokens)
        
        np.testing.assert_array_almost_equal(vllm_logprobs, hf_logprobs)
        
         
def _get_hf_logprobs(model_name: str, query: List[int],
                     response: List[int],
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"q "**********"u "**********"e "**********"r "**********"y "**********"_ "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********", "**********"  "**********"a "**********"t "**********"t "**********"e "**********"n "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"m "**********"a "**********"s "**********"k "**********", "**********"  "**********"p "**********"o "**********"s "**********"i "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"i "**********"d "**********"s "**********", "**********"  "**********"p "**********"a "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********") "**********"  "**********"- "**********"> "**********"  "**********"L "**********"i "**********"s "**********"t "**********"[ "**********"f "**********"l "**********"o "**********"a "**********"t "**********"] "**********": "**********"
    """Get logprobs using HuggingFace transformers."""
    padding_mask = "**********"= pad_token_id
    input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
    
    model: PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, DTYPE),
        device_map='cuda',
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    with torch.no_grad():
        input_ids = input_ids[None, :].to('cuda')
        attention_mask = attention_mask[None, :].to('cuda')
        position_ids = position_ids[None, :].to('cuda')
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits.to(torch.float32)
        logprobs = log_softmax_and_gather(logits, input_ids[:, 1:])
        logprobs = logprobs[:, len(query) - 1:]
    return logprobs.flatten().tolist()


def _get_vllm_logprobs(model_name: str, prompt: str) -> Dict[str, Union[List[str], List[int], List[float]]]:
    """Get logprobs using vLLM."""
    # Determine dtype based on model
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=True,  # Disable CUDA graph for consistency
        max_model_len=1024,
        dtype=DTYPE,
        disable_cascade_attn=True,
        gpu_memory_utilization=0.5,
    )
    
    sampling_params = vllm.SamplingParams(
        max_tokens= "**********"
        logprobs=0,  # Return top-1 logprob
        seed=SEED
    )
    
    # Generate
    outputs = "**********"=[prompt], sampling_params=sampling_params)
    output = outputs[0]
    
    # Extract logprobs
    response = []
    logprobs = []
    
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"n "**********"f "**********"o "**********"  "**********"i "**********"n "**********"  "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********". "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********"s "**********"[ "**********"0 "**********"] "**********". "**********"l "**********"o "**********"g "**********"p "**********"r "**********"o "**********"b "**********"s "**********": "**********"
        # Get the token and its logprob
        token_id = "**********"
        logprob_info = "**********"
        
        response.append(token_id)
        logprobs.append(logprob_info.logprob)
        
    return {
        "response": response,
        "logprobs": logprobs
    }


if __name__ == "__main__":
    unittest.main()