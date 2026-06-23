import os
import re
import ast
import numpy as np
import plotly.graph_objects as go

def clean_and_parse_list(buffer_lines):
    full_str = " ".join(buffer_lines).strip()
    if not full_str.endswith(']'):
        full_str += "]"
    try:
        parsed_list = ast.literal_eval(full_str)
        if isinstance(parsed_list, list):
            return parsed_list
    except:
        stripped = full_str.strip("[]")
        return [t.strip("'\" ") for t in stripped.split(',')]
    return []

def parse_peak_tokens_by_scan(txt_path, target_prompt_id, num_layers, intermediate_size):
    """
    Scans sequentially through the file looking for 'PROMPT X'.
    Reconstructs and collects the tokens dynamically per layer.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Could not find log file: '{txt_path}'")
        
    print(f"Scanning '{txt_path}' line-by-line for Prompt ID {target_prompt_id}...")
    
    token_grid = [None] * num_layers
    target_header = f"--- PROMPT {target_prompt_id}:"
    
    found_prompt = False
    current_layer_idx = None
    accumulated_buffer = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if target_header in line:
                found_prompt = True
                continue
                
            if found_prompt and line.startswith("--- PROMPT"):
                break
                
            if found_prompt:
                layer_match = re.match(r'^\s*Layer\s+(\d+)\s+Peak\s+Tokens:\s*(.*)', line)
                if layer_match:
                    if current_layer_idx is not None:
                        token_grid[current_layer_idx] = clean_and_parse_list(accumulated_buffer)
                    current_layer_idx = int(layer_match.group(1))
                    accumulated_buffer = [layer_match.group(2).strip()]
                else:
                    if current_layer_idx is not None and line.strip():
                        accumulated_buffer.append(line.strip())

        if current_layer_idx is not None:
            token_grid[current_layer_idx] = clean_and_parse_list(accumulated_buffer)

    if not found_prompt:
        raise ValueError(f"Prompt ID {target_prompt_id} was not found inside '{txt_path}'.")

    for i in range(num_layers):
        if token_grid[i] is None or len(token_grid[i]) == 0:
            token_grid[i] = [f"L{i}_N"] * intermediate_size
        elif len(token_grid[i]) < intermediate_size:
            diff = intermediate_size - len(token_grid[i])
            token_grid[i].extend(["[Truncated]"] * diff)
        elif len(token_grid[i]) > intermediate_size:
            token_grid[i] = token_grid[i][:intermediate_size]
            
    return token_grid

def generate_interactive_heatmap_with_slider(npz_path, txt_path, target_prompt_id):
    if not os.path.exists(npz_path):
        print(f"Error: File '{npz_path}' not found.")
        return

    # 1. Load data from the NPZ file
    data = np.load(npz_path)
    neurons = data['neurons']
    prompt = data['prompt']
    tag_desc = data['tag_description']
    num_layers, intermediate_size = neurons.shape

    # 2. Extract corresponding token strings from text logs dynamically
    token_grid = parse_peak_tokens_by_scan(txt_path, target_prompt_id, num_layers, intermediate_size)

    # 3. Layer-wise Percentile-Clipped Min-Max Scaling (2% to 98% Percentiles)
    print("Applying Normalization...")
    normalized_neurons = np.zeros_like(neurons)
    for i in range(num_layers):
        layer_data = neurons[i]
        q_min = np.percentile(layer_data, 2)
        q_max = np.percentile(layer_data, 98)
        denom = (q_max - q_min) if (q_max - q_min) > 1e-8 else 1e-8
        normalized_neurons[i] = np.clip((layer_data - q_min) / denom, 0.0, 1.0)

    # 4. Precompute the tooltips matrix
    hover_text = []
    for layer in range(num_layers):
        layer_hovers = []
        for neuron in range(intermediate_size):
            raw_val = neurons[layer, neuron]
            norm_val = normalized_neurons[layer, neuron]
            tok = token_grid[layer][neuron]
            
            hover_info = (
                f"<b>Layer:</b> {layer}<br>"
                f"<b>Neuron ID:</b> {neuron}<br>"
                f"<b>Trigger Token:</b> '{tok}'<br>"
                f"<b>Raw Activation:</b> {raw_val:.4f}<br>"
                f"<b>Robust Quantile Strength:</b> {norm_val:.4f}"
            )
            layer_hovers.append(hover_info)
        hover_text.append(layer_hovers)

    # 5. Set up Thresholding Configurations for the Slider Animation Steps
    print("Pre-calculating threshold frames for the interactive slider...")
    thresholds = np.linspace(0.0, 1.0, 11)
    frames_data = []
    
    for thresh in thresholds:
        thresholded_z = np.copy(normalized_neurons)
        # Any value falling under the threshold is masked out using NaN
        thresholded_z[thresholded_z < thresh] = np.nan
        frames_data.append(thresholded_z)

    # Custom high-contrast cold-to-hot colorscale explicitly defined here
    # Deep Navy Blue -> Muted Dark Charcoal Gray -> Bright Glowing Red
    custom_blue_red = [
        [0.0, "rgb(10, 40, 120)"],
        [0.5, "rgb(35, 35, 40)"],
        [1.0, "rgb(255, 0, 50)"]
    ]

    # 6. Initialize Figure with the default 0.0 threshold state (FIXED HERE)
    base_heatmap = go.Heatmap(
        z=frames_data[0],  # Passed single initial frame array slice
        x=list(range(intermediate_size)),
        y=list(range(num_layers)),
        text=hover_text,
        hoverinfo="text",
        colorscale=custom_blue_red,  
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(
            title=dict(
                text="Percentile-Clipped Min-Max Scale",
                side="right"
            )
        )
    )

    fig = go.Figure(data=[base_heatmap])

    # 7. Construct the Slider GUI Controller Array
    sliders_steps = []
    for idx, thresh in enumerate(thresholds):
        step = dict(
            method="restyle",
            label=f"{thresh:.1f}",
            args=[{"z": [frames_data[idx]]}]  
        )
        sliders_steps.append(step)

    sliders_config = [dict(
        active=0,
        currentvalue={"prefix": "<b>Activation Threshold Filter: </b> Only values above "},
        pad={"t": 60},
        steps=sliders_steps
    )]

    # 8. Render Global Layout Configurations 
    fig.update_layout(
        template="plotly_dark",                         
        paper_bgcolor="rgb(10, 10, 10)",                
        plot_bgcolor="rgb(0, 0, 0)",                    
        title={
            'text': f"<b>Interactive Neuron Heatmap (Max Token per Prompt)</b><br><span style='font-size:13px; color:#A0A0A0;'>Prompt: \"{prompt}\" | Category: {tag_desc}</span>",
            'y': 0.96, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title=f"Neuron Index",
        yaxis_title="Layer Index",
        xaxis=dict(showticklabels=True, nticks=15, gridcolor="rgb(25, 25, 25)"), 
        yaxis=dict(dtick=1, autorange="reversed", gridcolor="rgb(25, 25, 25)"),
        width=1500,
        height=900,
        sliders=sliders_config
    )

    # 9. Save out isolated HTML dashboard
    output_dir = "Prompts_Heatmap"
    os.makedirs(output_dir, exist_ok=True)
    output_html_name = os.path.join(output_dir, f"prompt_{target_prompt_id}_heatmap.html")
    fig.write_html(output_html_name, include_plotlyjs="cdn")
    print(f"Interactive Heatmap Generated: '{output_html_name}'")

if __name__ == "__main__":
    PROMPT_ID = 0
    TARGET_NPZ = f"activation_data_hooks/prompt_{PROMPT_ID}.npz"
    NEURON_TXT_LOG = "neuron_peak_tokens.txt"
    
    generate_interactive_heatmap_with_slider(TARGET_NPZ, NEURON_TXT_LOG, target_prompt_id=PROMPT_ID)
