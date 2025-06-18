#date: 2025-06-18T16:54:49Z
#url: https://api.github.com/gists/bddb42bbe5792237f61aae20b80833e1
#owner: https://api.github.com/users/903124

#Adapt from github.com/HKUNLP/Dream and github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time
import re
import traceback
import copy

# --- Outlines Imports ---
from outlines.processors.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Model Configurations ---
MODELS = {
    "LLaDA-8B-Instruct": {
        "path": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_id": 126336,
        "mask_token": "**********"
        "type": "llada"
    },
    "Dream-v0-Instruct-7B": {
        "path": "Dream-org/Dream-v0-Instruct-7B",
        "mask_id": "**********"
        "mask_token": "**********"
        "type": "dream"
    }
}

# --- Global State ---
model = None
tokenizer = "**********"
outlines_tokenizer = "**********"
current_model_name = ""
MASK_ID = -1
MASK_TOKEN = "**********"

# --- Model Loading Function ---
def load_model_and_tokenizer(model_name: "**********":
    """Loads the selected model and tokenizer and initializes components."""
    global model, tokenizer, outlines_tokenizer, current_model_name, MASK_ID, MASK_TOKEN

    if model_name == current_model_name:
        return

    print(f"\n--- Loading model: {model_name} ---")
    model_config = MODELS[model_name]
    model_path = model_config["path"]

    try:
        new_tokenizer = "**********"=True)
        print("Tokenizer loaded.")
        
        print("Loading model...")
        new_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).to(device)
        print("Model loaded and moved to device.")

        # Assign to globals only after successful loading
        tokenizer = "**********"
        model = new_model
        
        # Create Outlines-compatible tokenizer
        outlines_tokenizer = "**********"
        
        # Determine mask token ID based on model type
        if model_name == "Dream-v0-Instruct-7B":
            MASK_ID = "**********"
            if MASK_ID == -100:
                print("Warning: "**********"
        else:  # LLaDA-8B-Instruct
            MASK_ID = model_config["mask_id"]
        
        MASK_TOKEN = "**********"

        current_model_name = model_name
        print(f"Successfully loaded {model_name}. Mask ID: {MASK_ID}")

    except Exception as e:
        print(f"FATAL: Failed to load model {model_name}: {e}")
        traceback.print_exc()
        # Reset globals to prevent using a partially loaded model
        model = None
        tokenizer = "**********"
        outlines_tokenizer = "**********"
        current_model_name = ""
        raise gr.Error(f"Failed to load model {model_name}. Check console for details.")

def get_fsm_state_for_prefix(guide: "**********": list[int], cache: dict):
    """Calculates the FSM state for a given prefix of token IDs, using a cache."""
    prefix_tuple = "**********"
    if prefix_tuple in cache:
        return cache[prefix_tuple]

    current_state = guide.initial_state
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"p "**********"r "**********"e "**********"f "**********"i "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********"_ "**********"l "**********"i "**********"s "**********"t "**********": "**********"
        current_state = "**********"
    
    cache[prefix_tuple] = current_state
    return current_state

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part: continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    return constraints

def format_chat_history(history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

def add_gumbel_noise(logits, temperature):
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log(-torch.log(noise)) * temperature
    return logits + gumbel_noise

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"n "**********"u "**********"m "**********"_ "**********"t "**********"r "**********"a "**********"n "**********"s "**********"f "**********"e "**********"r "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"m "**********"a "**********"s "**********"k "**********"_ "**********"i "**********"n "**********"d "**********"e "**********"x "**********", "**********"  "**********"s "**********"t "**********"e "**********"p "**********"s "**********") "**********": "**********"
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = "**********"=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : "**********"
    return num_transfer_tokens

def unified_generate_with_visualization(model, tokenizer_hf, device, messages, gen_length= "**********"=32,
                                      constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                      remasking='low_confidence', model_type="llada",
                                      current_regex_guide=None, current_fsm_cache=None):
    """Unified generation function that works for both LLaDA and Dream models"""
    if constraints is None:
        constraints = {}
    
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = "**********"=False)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"e "**********"n "**********"u "**********"m "**********"e "**********"r "**********"a "**********"t "**********"e "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
            processed_constraints[pos + i] = "**********"
    
    chat_input = "**********"=True, tokenize=False)
    input_ids = "**********"
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    prompt_length = input_ids.shape[1]
    
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    visualization_states = []
    initial_state = "**********"
    visualization_states.append(initial_state)
    
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"p "**********"o "**********"s "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"p "**********"r "**********"o "**********"c "**********"e "**********"s "**********"s "**********"e "**********"d "**********"_ "**********"c "**********"o "**********"n "**********"s "**********"t "**********"r "**********"a "**********"i "**********"n "**********"t "**********"s "**********". "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********") "**********": "**********"
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[: "**********"
    
    prompt_index = (x != MASK_ID)
    
    if block_length > gen_length:
        block_length = gen_length
    
    num_blocks = (gen_length + block_length - 1) // block_length
    
    steps_per_block = steps // num_blocks if num_blocks > 0 else steps
    if steps_per_block < 1:
        steps_per_block = 1
    
    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        
        block_mask_indices_in_x = (x[:, block_start:block_end] == MASK_ID)
        
        if not block_mask_indices_in_x.any() and not processed_constraints:
            continue

        num_transfer_tokens_schedule = "**********"
        
        for step_idx in range(steps_per_block):
            mask_index_full_seq = (x == MASK_ID)
            if not mask_index_full_seq.any() and not processed_constraints:
                break

            # Get model logits
            try:
                with torch.no_grad():
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = MASK_ID
                        x_ = torch.cat([x, un_x], dim=0)
                        
                        # Forward pass through model
                        if model_type == "dream":
                            outputs = model(x_)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        else:
                            logits = model(x_).logits
                        
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        # Single forward pass
                        if model_type == "dream":
                            outputs = model(x)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        else:
                            logits = model(x).logits
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                traceback.print_exc()
                raise
            
            x0 = torch.zeros_like(x)

            # Apply structured generation if regex guide is available
            if current_regex_guide and current_fsm_cache is not None:
                x0_so_far = x.clone()

                for j_pos in range(block_start, block_end):
                    if x[0, j_pos] == MASK_ID:
                        prefix_token_ids = "**********"
                        for k_idx in range(prompt_length, j_pos):
                            token_at_k = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"a "**********"t "**********"_ "**********"k "**********"  "**********"! "**********"= "**********"  "**********"M "**********"A "**********"S "**********"K "**********"_ "**********"I "**********"D "**********": "**********"
                                prefix_token_ids.append(token_at_k)
                        
                        fsm_state = "**********"
                        instruction = current_regex_guide.get_next_instruction(fsm_state)
                        allowed_tokens = "**********"=torch.long)
                        
                        logits_for_pos = logits[0, j_pos].clone()
                        mask = torch.ones_like(logits_for_pos, dtype=torch.bool)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"l "**********"l "**********"o "**********"w "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********". "**********"n "**********"u "**********"m "**********"e "**********"l "**********"( "**********") "**********"  "**********"> "**********"  "**********"0 "**********": "**********"
                            mask[allowed_tokens] = "**********"
                        
                        logits_for_pos.masked_fill_(mask, -float('inf'))
                        
                        noisy_logits = add_gumbel_noise(logits_for_pos, temperature)
                        next_token_id = "**********"
                        x0_so_far[0, j_pos] = "**********"
                
                x0 = x0_so_far
            else:
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else: # 'random'
                x0_p = torch.rand_like(logits[:, :, 0])
            
            x0_p[:, :block_start] = -float('inf')
            x0_p[:, block_end:] = -float('inf')
            
            old_x = x.clone()
            x0_final = torch.where(mask_index_full_seq, x0, x)

            confidence = torch.where(mask_index_full_seq, x0_p, -float('inf'))
            confidence_block_scoped = confidence.clone()
            confidence_block_scoped[:, :block_start] = -float('inf')
            confidence_block_scoped[:, block_end:] = -float('inf')

            transfer_index = torch.zeros_like(x0_final, dtype=torch.bool)
            for j_batch_idx in range(confidence_block_scoped.shape[0]):
                masked_positions_in_block_confidence = confidence_block_scoped[j_batch_idx, block_start:block_end].clone()
                masked_positions_in_block_confidence[~block_mask_indices_in_x[j_batch_idx]] = -float('inf')

                k_to_transfer = "**********"
                                     (masked_positions_in_block_confidence > -float('inf')).sum().item())

                if k_to_transfer > 0:
                    if step_idx < steps_per_block - 1:
                        _, select_indices_in_block = torch.topk(masked_positions_in_block_confidence, k=k_to_transfer)
                        select_indices_global = select_indices_in_block + block_start
                        transfer_index[j_batch_idx, select_indices_global] = True
                    else:
                        transfer_index[j_batch_idx, block_start:block_end] = block_mask_indices_in_x[j_batch_idx]

            x = torch.where(transfer_index, x0_final, x)
            
         "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"p "**********"o "**********"s "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"p "**********"r "**********"o "**********"c "**********"e "**********"s "**********"s "**********"e "**********"d "**********"_ "**********"c "**********"o "**********"n "**********"s "**********"t "**********"r "**********"a "**********"i "**********"n "**********"t "**********"s "**********". "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********") "**********": "**********"
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[: "**********"
            
            current_state_vis = []
            for i_vis in range(gen_length):
                pos_vis = prompt_length + i_vis
                token_val, old_token_val = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"v "**********"a "**********"l "**********"  "**********"= "**********"= "**********"  "**********"M "**********"A "**********"S "**********"K "**********"_ "**********"I "**********"D "**********": "**********"
                    current_state_vis.append((MASK_TOKEN, "#444444"))
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"o "**********"l "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"v "**********"a "**********"l "**********"  "**********"= "**********"= "**********"  "**********"M "**********"A "**********"S "**********"K "**********"_ "**********"I "**********"D "**********": "**********"
                    token_str = "**********"=True)
                    confidence_val = float(x0_p[0, pos_vis].cpu()) if x0_p[0, pos_vis] != -float('inf') else 0.0
                    color = "#FF6666" if confidence_val < 0.3 else "#FFAA33" if confidence_val < 0.7 else "#66CC66"
                    current_state_vis.append((token_str, color))
                else:
                    token_str = "**********"=True)
                    current_state_vis.append((token_str, "#6699CC"))
            
            visualization_states.append(current_state_vis)
    
    response_tokens_ids = [tid for tid in x[0, prompt_length: "**********"
    
    final_text = "**********"
        response_tokens_ids, 
        skip_special_tokens= "**********"
        clean_up_tokenization_spaces= "**********"
    )
    
    return visualization_states, final_text

css = """
.category-legend{display:none}
button{height: 60px}
"""

def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Unified Diffusion Model Demo with Structured Generation")
        gr.Markdown("### A unified interface for LLaDA and Dream-v0 with Outlines regex guidance.")
        
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()), 
                value=list(MODELS.keys())[0], 
                label="Select Model",
                info="The selected model will be loaded on first generation."
            )
            
        chat_history = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500)
                with gr.Group():
                    with gr.Row():
                        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here or select an example...", show_label=False)
                        send_btn = gr.Button("Send")
                constraints_input = gr.Textbox(label="Word Constraints (Positional)", info="Format: '0:Word, 5:Another'", placeholder="0:Once, 5:upon, 10:time")
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(label="Denoising Process Visualization", combine_adjacent=False, show_legend=True)

        with gr.Accordion("Regex Constraint", open=True):
            regex_input = gr.Textbox(
                label="Regex Pattern",
                value=r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
                info="The model will be constrained to generate text matching this regex. Try an example!",
                interactive=True
            )
            gr.Examples(
                examples=[
                    [r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)", "Generate a random IP address."],
                    [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "The new user's email address is "],
                    [r"\d{4}-\d{2}-\d{2}", "Invoice date (YYYY-MM-DD): "],
                    [r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "The project's documentation is available at "],
                    [r'\{\s*"user":\s*".+",\s*"id":\s*\d+\s*\}', 'Create a JSON object for user "test" with id 123. '],
                ],
                inputs=[regex_input, user_input],
                label="Regex Examples",
            )

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(minimum=16, maximum=128, value=64, step=8, label="Generation Length")
                steps = gr.Slider(minimum=8, maximum=64, value=32, step=4, label="Denoising Steps")
            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
                cfg_scale = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale")
            with gr.Row():
                block_length = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Block Length")
                remasking_strategy = gr.Radio(choices=["low_confidence", "random"], value="low_confidence", label="Remasking Strategy")
            with gr.Row():
                visualization_delay = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, label="Visualization Delay (seconds)")

        current_response = gr.Textbox(label="Current Response", lines=3, visible=False)
        clear_btn = gr.Button("Clear Conversation")
        
        def add_message(history, message, response):
            return history + [[message, response]]
            
        def user_message_submitted(message, history):
            if not message.strip():
                return history, history, "", [], ""
            history = add_message(history, message, None)
            return history, history, "", [], ""
            
        def bot_response_stream(model_name, history, regex_str, gen_len, num_steps, const_text, delay, temp, cfg, block_len, remask):
            try:
                load_model_and_tokenizer(model_name)
            except Exception as e:
                error_msg = f"Failed to load model: {e}"
                if history:
                    history[-1][1] = error_msg
                else:
                    history = [["System", error_msg]]
                yield history, [(error_msg, "red")], error_msg
                return

            if not history or history[-1][1] is not None:
                yield history, [], ""
                return
            
            # Initialize regex guide for this specific generation
            current_regex_guide = None
            fsm_cache = {}
            if regex_str:
                try:
                    current_regex_guide = "**********"
                    print(f"Successfully created RegexGuide for pattern: {regex_str}")
                except Exception as e:
                    error_msg = f"Invalid Regex Pattern: {e}"
                    print(error_msg)
                    history[-1][1] = error_msg
                    yield history, [(error_msg, "red")], error_msg
                    return

            messages = format_chat_history(history)
            
            try:
                model_config = MODELS[model_name]
                model_type = model_config["type"]
                
                parsed_constraints = parse_constraints(const_text)
                vis_states, response_text = unified_generate_with_visualization(
                    model, tokenizer, device, messages, 
                    gen_length=gen_len, steps=num_steps, constraints=parsed_constraints,
                    temperature=temp, cfg_scale=cfg, block_length=block_len, remasking=remask,
                    model_type=model_type,
                    current_regex_guide=current_regex_guide, current_fsm_cache=fsm_cache
                )
                
                history[-1][1] = response_text
                
                if not vis_states:
                    yield history, [], response_text
                    return

                yield history, vis_states[0], response_text
                for state in vis_states[1:]:
                    time.sleep(delay)
                    yield history, state, response_text
                    
            except Exception as e:
                error_msg = f"Error during generation: {e}"
                print(error_msg)
                traceback.print_exc()
                history[-1][1] = error_msg
                yield history, [(error_msg, "red")], error_msg
        
        def clear_conversation():
            return [], [], "", []
        
        clear_btn.click(fn=clear_conversation, outputs=[chat_history, chatbot_ui, current_response, output_vis])
        
        trigger_args = {
            "fn": user_message_submitted,
            "inputs": [user_input, chat_history],
            "outputs": [chat_history, chatbot_ui, user_input, output_vis, current_response]
        }
        response_args = {
            "fn": bot_response_stream,
            "inputs": [model_selector, chat_history, regex_input, gen_length, steps, constraints_input, visualization_delay, temperature, cfg_scale, block_length, remasking_strategy],
            "outputs": [chatbot_ui, output_vis, current_response]
        }

        user_input.submit(**trigger_args).then(**response_args)
        send_btn.click(**trigger_args).then(**response_args)
        
    return demo

if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(share=True)