import os
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global dictionary to transiently hold intermediate MLP activations per token
mlp_activations = {}

TAG_DESCRIPTIONS = {
    0: "Countries and Cities", 1: "Animals", 2: "Bands", 3: "Emotions",
    4: "Equations", 5: "Famous People", 6: "Colors", 7: "Trees",
    8: "Flowers", 9: "Famous Dishes", 10: "Fictional Characters",
    11: "Famous Quotes", 12: "Famous Landmarks", 13: "Mythical Creatures",
    14: "Famous Movies", 15: "Famous Books", 16: "Diseases",
}

def make_mlp_hook(layer_index):
    """
    Creates a hook function for the input of down_proj.
    Keeps input[0] intact to isolate the exact activation tensor.
    """
    def hook(module, input, output):
        # Explicitly extract the tensor from the tuple to maintain data continuity
        mlp_activations[layer_index] = input[0].detach().to(torch.float32).cpu()
    return hook

def main_process_prompt(promt_in_line, prompt_id, tokenizer, model, f_peaks, OUTPUT_DIR_NPZ, TOTAL_HEADS, NUM_LAYERS, INTERMEDIATE_SIZE): 
    # Split the incoming string and clean whitespace safely
    parts = promt_in_line.split(',')
    PROMPT = parts[0].strip()  
    try:
        tag_int = int(parts[1].strip())
        TAG = str(tag_int)
        TAG_DESC = TAG_DESCRIPTIONS.get(tag_int, "Unknown Category")
    except (IndexError, ValueError):
        TAG = "None"
        TAG_DESC = "Unknown Category"

    # 1. Run inference
    inputs = tokenizer(PROMPT, return_tensors="pt")
    num_prompt_tokens = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Clear global cache before forward pass
    mlp_activations.clear()

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_texts = [all_tokens[i].replace("Ġ", " ") for i in range(num_prompt_tokens)]

    # 2. Initialize Accumulators for the High-Density NumPy Matrices
    aggregated_neurons = np.zeros((NUM_LAYERS, INTERMEDIATE_SIZE), dtype=np.float32)
    aggregated_heads = np.zeros((len(attentions), TOTAL_HEADS), dtype=np.float32)  

    # 3. Process Peak Max Neurons and identify triggering tokens
    f_peaks.write(f"--- PROMPT {prompt_id}: '{PROMPT}' (Tag: {TAG} - {TAG_DESC}) ---\n")
    
    for layer_index in range(NUM_LAYERS):
        # Extract the tensor safely from the global dict
        layer_tokens_tensor = mlp_activations[layer_index][0, :, :] # [seq_len, intermediate_size]

        max_activation_per_neuron, max_token_indices = layer_tokens_tensor.max(dim=0)
        aggregated_neurons[layer_index] = max_activation_per_neuron.numpy()

        max_token_indices_list = max_token_indices.tolist()
        layer_peak_tokens = [token_texts[idx] for idx in max_token_indices_list]
        
        f_peaks.write(f"  Layer {layer_index} Peak Tokens: {layer_peak_tokens}\n")
    f_peaks.write("\n")

    # 4. Export Attention Heads (Averages)
    total_attention_counts = np.zeros((len(attentions), TOTAL_HEADS), dtype=np.float32)

    for token_index in range(num_prompt_tokens):
        if token_index > 0:
            for layer_index, layer_attention_tensor in enumerate(attentions):
                token_attention_matrix = layer_attention_tensor[0, :, token_index, :token_index+1]
                mean_per_head = torch.mean(token_attention_matrix.to(torch.float32), dim=1).cpu().numpy()
                
                aggregated_heads[layer_index, :] += mean_per_head
                total_attention_counts[layer_index, :] += 1
    
    # Divide out the accumulated steps to get a pure mathematical average across tokens
    total_attention_counts[total_attention_counts == 0] = 1
    aggregated_heads = aggregated_heads / total_attention_counts

    # 5. Save uniform data as a compressed NumPy file ready for downstream clustering
    npz_filename = os.path.join(OUTPUT_DIR_NPZ, f"prompt_{prompt_id}.npz")
    np.savez_compressed(
        npz_filename,
        neurons=aggregated_neurons, 
        attentions=aggregated_heads,
        prompt=PROMPT,
        tag=TAG,
        tag_description=TAG_DESC
    )

if __name__ == "__main__":
    #MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M" # Removed "-Instruct"
    OUTPUT_DIR_NPZ = "activation_data_hooks"
    os.makedirs(OUTPUT_DIR_NPZ, exist_ok=True)

    print("Loading tokenizer and model into memory...")
    global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    device = "cpu"
    torch_dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"🚀 Accelerating execution using CUDA GPU ({torch_dtype})...")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Accelerating execution using Apple Silicon MPS...")

    global_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        output_attentions=True,
        torch_dtype=torch_dtype
    ).to(device)
    global_model.eval()

    TOTAL_HEADS = global_model.config.num_attention_heads
    NUM_LAYERS = global_model.config.num_hidden_layers 
    INTERMEDIATE_SIZE = global_model.config.intermediate_size

    # Register Forward Hooks to all MLP sub-layers
    hook_handles = []
    for i in range(NUM_LAYERS):
        target_layer = global_model.model.layers[i].mlp.down_proj
        handle = target_layer.register_forward_hook(make_mlp_hook(layer_index=i))
        hook_handles.append(handle)
    print(f"🎯 Successfully wiretapped {NUM_LAYERS} MLP layers (Tracking {INTERMEDIATE_SIZE} true neurons each).")

    # Parse dataset entries securely
    promtp_list = []
    if os.path.exists("prompts.txt"):
        print("Reading and categorizing your prompt source data...")
        with open("prompts.txt", "r", encoding="utf-8") as prompts:
            tag = -1
            for line in prompts:
                if '--' in line:
                    tag = tag + 1
                else:
                    if line.strip():  
                        promtp_list.append(line.strip() + ',' + str(int(tag)))
    else:
        print("Warning: prompts.txt file was not found.")

    OUTPUT_FILE_PEAKS = "neuron_peak_tokens.txt"     

    if promtp_list:
        print(f"Starting high-speed feature extraction for {len(promtp_list)} prompts...")
        with open(OUTPUT_FILE_PEAKS, "w", encoding="utf-8") as f_peaks:
             
            f_peaks.write(
                f"=== LLM NEURON PEAK TRIGGER TOKENS ===\n"
                f"Model: {MODEL_NAME}\n\n"
            )

            for idx, item in enumerate(promtp_list):
                main_process_prompt(
                    promt_in_line=item,
                    prompt_id=idx,
                    tokenizer=global_tokenizer,
                    model=global_model,
                    f_peaks=f_peaks,
                    OUTPUT_DIR_NPZ=OUTPUT_DIR_NPZ,
                    TOTAL_HEADS=TOTAL_HEADS,
                    NUM_LAYERS=NUM_LAYERS,
                    INTERMEDIATE_SIZE=INTERMEDIATE_SIZE
                )
                
                # Print explicit updates so you can track performance
                if idx % 50 == 0 or idx == len(promtp_list) - 1:
                    print(f"Processed {idx + 1}/{len(promtp_list)} prompts...")
                    f_peaks.flush() 
                    
                    if device == "mps":
                        torch.mps.empty_cache()
                    gc.collect()

        print("Feature extraction complete. Saved max neuron activations per prompt successfully.")
