
import os
import re
import ast
import numpy as np
import plotly.graph_objects as go

TAG_DESCRIPTIONS = {
    0: "Countries and Cities", 1: "Animals", 2: "Bands", 3: "Emotions",
    4: "Equations", 5: "Famous People", 6: "Colors", 7: "Trees",
    8: "Flowers", 9: "Famous Dishes", 10: "Fictional Characters",
    11: "Famous Quotes", 12: "Famous Landmarks", 13: "Mythical Creatures",
    14: "Famous Movies", 15: "Famous Books", 16: "Diseases",
}

def pipeline_aggregate_with_sharing_hovers(npz_dir, consensus_threshold=0.90, leakage_threshold=0.05): # consensus and leakage default values
    if not os.path.exists(npz_dir):
        print(f"Error: Directory '{npz_dir}' not found.")
        return

    files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
    if not files:
        print(f"No .npz targets discovered inside '{npz_dir}'.")
        return

    sample = np.load(files[0])
    num_layers, intermediate_size = sample['neurons'].shape
    print(f"Processing raw data: {len(files)} files discovered. Width: {intermediate_size} neurons.")

    tag_accumulators = {tag_id: [] for tag_id in TAG_DESCRIPTIONS.keys()}

    # --- STEP 1: Process, Robust Scale, and Binarize ---
    print("Executing Step 1: Performing percentile-clipping and min-max Scaling and Binarization...")
    for f_path in files:
        try:
            data = np.load(f_path)
            raw_neurons = data['neurons']
            tag_id = int(str(data['tag']).strip())
            
            if tag_id not in tag_accumulators:
                continue
                
            binary_matrix = np.zeros_like(raw_neurons, dtype=np.uint8)
            for layer in range(num_layers):
                layer_data = raw_neurons[layer]
                q_min = np.percentile(layer_data, 2)
                q_max = np.percentile(layer_data, 98)
                denom = (q_max - q_min) if (q_max - q_min) > 1e-8 else 1e-8
                
                norm_layer = np.clip((layer_data - q_min) / denom, 0.0, 1.0)
                binary_matrix[layer, norm_layer >= 1.0] = 1
                
            tag_accumulators[tag_id].append(binary_matrix)
        except Exception as e:
            print(f"Skipping malformed file {f_path}: {e}")

    # --- STEP 2: Compute Base Tag Consensus & Background Leakage ---
    print("\nExecuting Step 2: Compiling Cross-Prompt Category Overlaps...")
    base_consensus_masks = {}
    raw_frequency_ratios = {}
    
    for tag_id in sorted(TAG_DESCRIPTIONS.keys()):
        matrix_list = tag_accumulators[tag_id]
        if not matrix_list:
            continue
            
        stacked_tag_data = np.stack(matrix_list, axis=0)
        frequency_ratio = np.mean(stacked_tag_data, axis=0)
        raw_frequency_ratios[tag_id] = frequency_ratio
        
        consensus_mask = np.zeros((num_layers, intermediate_size), dtype=np.uint8)
        consensus_mask[frequency_ratio >= consensus_threshold] = 1
        base_consensus_masks[tag_id] = consensus_mask

    # --- STEP 3: Map Global Sharing Connections (FIXED LOGIC) ---
    print("\nExecuting Step 3: Mapping Multi-Tag Neural Sharing Matrices...")
    
    display_tag_masks = {}
    sharing_directory = [[[] for _ in range(intermediate_size)] for _ in range(num_layers)]
    
    # FIXED: Build the sharing directory using a low leakage threshold.
    # If a neuron fires even 5% of the time in another category, it is NOT unique.
    for layer in range(num_layers):
        for neuron in range(intermediate_size):
            for tag_id, freq_matrix in raw_frequency_ratios.items():
                if freq_matrix[layer, neuron] >= leakage_threshold:
                    sharing_directory[layer][neuron].append(tag_id)

    # Compute final display states
    for target_tag_id, target_mask in base_consensus_masks.items():
        display_matrix = np.copy(target_mask).astype(np.float32)
        
        for layer in range(num_layers):
            for neuron in range(intermediate_size):
                if target_mask[layer, neuron] == 1:
                    all_sharing_tags = sharing_directory[layer][neuron]
                    
                    # It is only unique if NO other tag triggers it above the leakage threshold
                    if len(all_sharing_tags) == 1 and target_tag_id in all_sharing_tags:
                        display_matrix[layer, neuron] = 2.0
                        
        display_tag_masks[target_tag_id] = display_matrix

    # --- STEP 4: Build Layout Tracks with Dynamic Hover Strings ---
    print("\nExecuting Step 4: Compiling Dynamic Tooltips and Layout Tracks...")
    fig = go.Figure()
    dropdown_buttons = []
    
    custom_specialization_scale = [
        [0.0, "rgb(10, 30, 80)"],    # 0 = Dark Navy Blue
        [0.5, "rgb(255, 0, 50)"],    # 1 = Bright Red
        [1.0, "rgb(255, 255, 255)"]  # 2 = Pure White
    ]

    active_trace_index = 0
    first_visible_tag_name = ""

    for tag_id in sorted(display_tag_masks.keys()):
        tag_name = TAG_DESCRIPTIONS[tag_id]
        display_matrix = display_tag_masks[tag_id]
        
        if first_visible_tag_name == "":
            first_visible_tag_name = tag_name

        hover_text = []
        for layer in range(num_layers):
            layer_hovers = []
            for neuron in range(intermediate_size):
                val = display_matrix[layer, neuron]
                
                sharing_ids = sharing_directory[layer][neuron]
                other_sharing_names = [TAG_DESCRIPTIONS[sid] for sid in sharing_ids if sid != tag_id]
                
                if val == 2.0:
                    status_str = "<span style='color:white;'><b>SPECIALIZED NEURON</b></span>"
                    sharing_details = "<i>None. This neuron is entirely exclusive to this tag.</i>"
                elif val == 1.0:
                    status_str = "<span style='color:red;'><b>SHARED CONSENSUS NEURON</b></span>"
                    bullets = "".join([f"<br> • {name}" for name in other_sharing_names])
                    sharing_details = f"<b>Also active in ({len(other_sharing_names)} other tags):</b>{bullets}"
                else:
                    status_str = "Inactive Background Void (0)"
                    sharing_details = "N/A"
                
                hover_info = (
                    f"<b>Active Tag:</b> {tag_name}<br>"
                    f"<b>Layer:</b> {layer}<br>"
                    f"<b>Neuron ID:</b> {neuron}<br>"
                    f"<b>State:</b> {status_str}<br><br>"
                    f"{sharing_details}"
                )
                layer_hovers.append(hover_info)
            hover_text.append(layer_hovers)

        fig.add_trace(go.Heatmap(
            z=display_matrix,
            x=list(range(intermediate_size)),
            y=list(range(num_layers)),
            text=hover_text,
            hoverinfo="text",
            colorscale=custom_specialization_scale,
            zmin=0.0,
            zmax=2.0,
            showscale=False,
            visible=(active_trace_index == 0),
            name=tag_name
        ))

        dropdown_buttons.append({
            "label": tag_name,
            "trace_idx": active_trace_index
        })
        active_trace_index += 1

    total_traces = active_trace_index
    plotly_buttons_config = []

    for btn in dropdown_buttons:
        trace_idx = btn["trace_idx"]
        tag_name = btn["label"]
        visibility_mask = [False] * total_traces
        visibility_mask[trace_idx] = True
        
        plotly_buttons_config.append(dict(
            method="update",
            label=tag_name,
            args=[
                {"visible": visibility_mask},
                {
                    "title.text": (
                        f"<b>LLM Semantic Fingerprint ({consensus_threshold*100:g}% Consensus | {leakage_threshold*100:g}% Leakage)</b><br>"
                        f"<span style='font-size:13px; color:#A0A0A0;'>Active Category: {tag_name} | Hover to inspect cross-tag sharing connections</span>"
                    )
                }
            ]
        ))

    updatemenus_config = [dict(
        type="dropdown",
        active=0,
        buttons=plotly_buttons_config,
        pad={"r": 10, "t": 10},
        x=0.0, xanchor="left", y=1.12, yanchor="top",
        bgcolor="rgb(20, 20, 25)", font=dict(color="white")
    )]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(10, 10, 10)",
        plot_bgcolor="rgb(0, 0, 0)",
        title={
            # FIXED: Title text is now completely dynamic based on your function arguments
            'text': (
                    f"<b>LLM Semantic Fingerprint ({consensus_threshold*100:g}% Consensus | {leakage_threshold*100:g}% Leakage)</b><br>"
                    f"<span style='font-size:13px; color:#A0A0A0;'>Active Category: {tag_name} | Hover to inspect cross-tag sharing connections</span>"
            ),
            'y': 0.96, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title=f"Neuron Index (0 to {intermediate_size - 1}) — {intermediate_size} Total Neurons",
        yaxis_title="Transformer Layer Index",
        xaxis=dict(showticklabels=True, nticks=15, gridcolor="rgb(20, 20, 20)"),
        yaxis=dict(dtick=1, autorange="reversed", gridcolor="rgb(20, 20, 20)"),
        width=1500,
        height=850,
        updatemenus=updatemenus_config
    )



    output_dir = "Concept_Heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"shared_networks_c{int(consensus_threshold*100)}_l{int(leakage_threshold*100)}_dashboard.html"
    output_html_name = os.path.join(output_dir, file_name)
    # Write the compiled dashboard file to disk
    fig.write_html(output_html_name, include_plotlyjs="cdn")
    print(f"Complete! Interactive dashboard compiled: '{output_html_name}'")

if __name__ == "__main__":
    DATA_DIR = "activation_data_hooks"
    pipeline_aggregate_with_sharing_hovers(DATA_DIR, consensus_threshold=0.5, leakage_threshold=0.25) #0.5; 0.25

  

