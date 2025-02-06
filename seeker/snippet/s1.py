#date: 2025-02-06T16:58:28Z
#url: https://api.github.com/gists/fa00ed81a8e8b973a823e3c01e82ffd6
#owner: https://api.github.com/users/903124

# outlines/processor/structured.py
...
class GuideLogitsProcessor(OutlinesLogitsProcessor):
    """Bias generation using a finite

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    guide
        The `outlines.fsm.Guide` which is used to bias the logits.
    """

    tokenizer: "**********"
    guide: Guide
    _guide_states: Dict[int, Any]
    _seq_start_idx: Optional[int]

    def __init__(self, tokenizer: "**********": Guide):
        """A Guide-based logits processor.

        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        guide
            The `outlines.fsm.Guide. which is used to bias the logits.
        """
        self.tokenizer = "**********"
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def process_logits(
        self, input_ids: torch.LongTensor, logits: torch.FloatTensor
    ) -> torch.Tensor:
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: List[int] = []

        for seq_ids in input_ids:
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        allowed_tokens_batch = "**********"
        batch_indices = []
        for i, guide_state in enumerate(sequence_states):
            instruction = self.guide.get_next_instruction(guide_state, input_ids[i:i+1])
            allowed_tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"l "**********"l "**********"o "**********"w "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"i "**********"s "**********"  "**********"n "**********"o "**********"t "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
                # Convert to LongTensor and ensure device matching
                allowed_tokens = "**********"=torch.long, device=logits.device)
                allowed_tokens_batch.append(allowed_tokens)
                batch_indices.append(torch.full_like(allowed_tokens, i, dtype= "**********"

        if not allowed_tokens_batch: "**********"
            return torch.full_like(logits, float("-inf"))

        allowed_tokens_concat = "**********"
        batch_indices_concat = torch.cat(batch_indices)

        # Create mask and ensure proper types
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[batch_indices_concat, allowed_tokens_concat] = "**********"
        logits.masked_fill_(mask, float("-inf"))

        return logits

    def copy(self) -> "GuideLogitsProcessor":
        """Return a copy of the logits processor."""
        return GuideLogitsProcessor(tokenizer= "**********"=self.guide.copy())
...

# outlines_core/fsm/guide.py

class EOSContinueRegexGuide(RegexGuide):
    """Guide that continues generation after EOS or when next state is None"""
    
    def __init__(
        self, 
        states_to_token_maps, 
        empty_token_ids, 
        eos_tensor, 
        initial_state,
        continue_text: str,
        max_tries: int = 1
    ):
        super().__init__(states_to_token_maps, empty_token_ids, eos_tensor, initial_state)
        self.continue_text = continue_text
        self.max_tries = max_tries
        self.current_tries = 0
        self._tokenizer = "**********"
        self.needs_continuation = False

    @classmethod
    def from_regex(cls, regex_string: "**********": str, max_tries: int = 1, device=None):
        states_to_token_maps, empty_token_ids, fsm_finals = "**********"
            regex_string, tokenizer
        )
        eos_tensor = "**********"=device)
        initial_state = "**********"
        
        guide = cls(
            states_to_token_maps, 
            empty_token_ids, 
            eos_tensor, 
            initial_state,
            continue_text=continue_text,
            max_tries=max_tries
        )
        guide._tokenizer = "**********"
        return guide

    def _try_continuation(self, input_ids: Optional[torch.Tensor] = None) -> Optional[Instruction]:
        """Handle continuation logic while preserving context"""
        if self.current_tries < self.max_tries:
            # Create new regex for continuation
            continue_regex = f"{re.escape(self.continue_text)}[A-Za-z\\s]+\\."
            
            # Create new FSM for continuation
            new_states_map, new_empty_ids, _ = create_states_mapping(
                continue_regex, 
                self._tokenizer
            )
            
            # Update guide state
            self.states_to_token_maps = "**********"
            self.empty_token_ids = "**********"
            self.initial_state = new_states_map.get_initial_state()
            self.current_tries += 1

            # Decode and store the generated text so far
            if input_ids is not None:
                self.generated_text = "**********"
            
            # Get continuation tokens
            continue_tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"c "**********"o "**********"n "**********"t "**********"i "**********"n "**********"u "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********", "**********"  "**********"t "**********"o "**********"r "**********"c "**********"h "**********". "**********"T "**********"e "**********"n "**********"s "**********"o "**********"r "**********") "**********": "**********"
                continue_tokens = "**********"
            
            # For the first token after completion, ONLY allow the first token of continue_text
            return Generate([continue_tokens[0]])

    def get_next_state(self, state: "**********": int) -> int:
        """Override to handle None next_state as continuation trigger"""
        if state == -1:
            return -1
            
        next_state = "**********"
        if next_state is None and self.current_tries < self.max_tries:
            # Reset to initial state for continuation
            return self.initial_state
        return -1 if next_state is None else next_state

    def get_next_instruction(self, state: int, input_ids: Optional[torch.Tensor] = None) -> Instruction:
        """Handle both EOS and None next_state cases with context"""
        if state == -1:
            continuation = self._try_continuation(input_ids)
            if continuation:
                return continuation
            return Write(self.eos_tensor)
            
        next_tokens_mask = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"e "**********"x "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"m "**********"a "**********"s "**********"k "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
            continuation = self._try_continuation(input_ids)
            if continuation:
                return continuation
            return Write(self.eos_tensor)
            
        return Generate(torch.tensor(next_tokens_mask))


    def copy(self):
        copied = EOSContinueRegexGuide(
            self.states_to_token_maps,
            self.empty_token_ids,
            self.eos_tensor,
            self.initial_state,
            self.continue_text,
            self.max_tries
        )
        copied.current_tries = self.current_tries
        copied._tokenizer = "**********"
        return copied    

#Inference      
      
import outlines
from transformers import AutoTokenizer
from outlines_core.fsm.guide import EOSContinueRegexGuide
from outlines.processors.structured import GuideLogitsProcessor

# 1. Using with LlamaCpp model
model = outlines.models.llamacpp(
    "microsoft/Phi-3-mini-4k-instruct-gguf", 
    "Phi-3-mini-4k-instruct-q4.gguf"
)


guide = EOSContinueRegexGuide.from_regex(
    regex_string=r"[A-Za-z\s]+\.",
    tokenizer= "**********"
    continue_text=" But wait,",
    max_tries=3
)

# Create processor with guide
processor = "**********"=model.tokenizer, guide=guide)

# Create generator and set processor
generator = outlines.generate.text(model)
generator.logits_processor = processor


# # Use the generator
prompt = """<|im_start|>system You are a helpful assistant.
<|im_end|>

<|im_start|>user
How many r in the word starberry?
<|im_end|>
<|im_start|>assistant"""
structured = "**********"=1000)
for chunk in structured:
    print(chunk, end="", flush=True)