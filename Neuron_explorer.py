import os
import re
import glob
import numpy as np
import plotly.graph_objects as go

# Configuration
DATA_DIR = "activation_data_hooks"
TXT_FILE = "neuron_peak_tokens.txt"

TAG_DESCRIPTIONS = {
    0: "Countries and Cities", 1: "Animals", 2: "Bands", 3: "Emotions",
    4: "Equations", 5: "Famous People", 6: "Colors", 7: "Trees",
    8: "Flowers", 9: "Famous Dishes", 10: "Fictional Characters",
    11: "Famous Quotes", 12: "Famous Landmarks", 13: "Mythical Creatures",
    14: "Famous Movies", 15: "Famous Books", 16: "Diseases",
}

# Color palette mapped to each TAG ID for visual contrast
COLOR_PALETTE = {
    0: "#1f77b4", 1: "#aec7e8", 2: "#ff7f0e", 3: "#ffbb78",
    4: "#2ca02c", 5: "#98df8a", 6: "#d62728", 7: "#ff9896",
    8: "#9467bd", 9: "#c5b0d5", 10: "#8c564b", 11: "#c49c94",
    12: "#e377c2", 13: "#f7b6d2", 14: "#7f7f7f", 15: "#c7c7c7",
    16: "#bcbd22"
}

def parse_txt_file(txt_path):
    """Parses the text file into a dictionary structured by Prompt ID and Layer."""
    prompt_data = {}
    current_prompt_id = None
    
    prompt_regex = re.compile(r"--- PROMPT (\d+):")
    layer_regex = re.compile(r"Layer (\d+)\s+Peak Tokens:\s+\[(.*)\]")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Could not find TXT file at {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            prompt_match = prompt_regex.search(line)
            if prompt_match:
                current_prompt_id = int(prompt_match.group(1))
                prompt_data[current_prompt_id] = {}
                continue
            
            if current_prompt_id is not None:
                layer_match = layer_regex.search(line)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    tokens_str = layer_match.group(2)
                    tokens = [t.strip().strip("'\"") for t in tokens_str.split(",") if t.strip()]
                    prompt_data[current_prompt_id][layer_num] = tokens
                    
    return prompt_data

def collect_neuron_data(target_layer, target_neuron_idx):
    """Gathers activations and tokens across all npz files for a specific neuron position."""
    print(" Reading text file token allocations into memory...")
    parsed_tokens = parse_txt_file(TXT_FILE)
    compiled_data = []
    
    npz_pattern = os.path.join(DATA_DIR, "prompt_*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in folder: '{DATA_DIR}'")

    print(f" Scanning {len(npz_files)} files. Please wait...")
    for idx, file_path in enumerate(npz_files):
        file_name = os.path.basename(file_path)
        match = re.search(r"prompt_(\d+)\.npz", file_name)
        if not match:
            continue
        prompt_id = int(match.group(1))
        
        with np.load(file_path, allow_pickle=True) as data:
            neurons_tensor = data['neurons']
            prompt_text = str(data['prompt'])
            tag_id = int(data['tag'])
            
            try:
                activation = float(neurons_tensor[target_layer, target_neuron_idx])
            except IndexError:
                continue
            
            token = "N/A"
            if prompt_id in parsed_tokens and target_layer in parsed_tokens[prompt_id]:
                layer_tokens = parsed_tokens[prompt_id][target_layer]
                if target_neuron_idx < len(layer_tokens):
                    token = layer_tokens[target_neuron_idx]

            compiled_data.append({
                "prompt_id": prompt_id,
                "activation": activation,
                "token": token,
                "prompt": prompt_text,
                "tag": tag_id,
                "tag_desc": TAG_DESCRIPTIONS.get(tag_id, f"Unknown Tag ({tag_id})"),
                "color": COLOR_PALETTE.get(tag_id, "#333333")
            })
            
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(npz_files)} files...")
            
    compiled_data.sort(key=lambda x: x["activation"])
    return compiled_data

def create_interactive_chart(layer, neuron_idx):
    """Generates the Plotly figure with permanent labels and custom click actions."""
    data_list = collect_neuron_data(layer, neuron_idx)
    
    if not data_list:
        print(f"No valid data matching Layer {layer}, Neuron {neuron_idx} was extracted.")
        return

    activations = [d["activation"] for d in data_list]
    tokens = [d["token"] for d in data_list]
    prompts = [d["prompt"] for d in data_list]
    colors = [d["color"] for d in data_list]
    
    # Using numerical sequential keys for x coordinates instead of overcrowded string lists
    x_coords = list(range(len(data_list)))

    # Extract Top 3 Activations (the final 3 elements in our ascending sorted data collection)
    top_3 = data_list[-3:][::-1]  # Reverse slice order to rank sequentially: 1st, 2nd, 3rd
    
    # Format structural typography card data using standard inline style blocks
    top_3_text = "<b>Top 3 Peak Activations</b><br>"
    for rank, d in enumerate(top_3, 1):
        top_3_text += (
            f"<br><b>#{rank} Token:</b> <span style='color:{d['color']}'><b>'{d['token']}'</b></span> "
            f"(Act: {d['activation']:.4f})<br>"
            f"<b>Tag:</b> {d['tag_desc']}<br>"
            f"<b>Prompt:</b> {d['prompt']}<br>"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=x_coords,
        y=activations,
        text=tokens,
        textposition='none', # Hides tokens from overcrowding bars statically
        marker_color=colors,
        hoverinfo="text",
        hovertext=[f"<b>Rank Count:</b> {i}<br><b>Prompt ID:</b> {d['prompt_id']}<br><b>Token:</b> {d['token']}<br><b>Activation:</b> {d['activation']:.4f}<br><b>{d['tag_desc']}</b><br><br><b>Hover Prompt:</b><br>{d['prompt']}" 
                   for i, d in enumerate(data_list)],
        customdata=prompts 
    ))

    # Add the Top 3 Summary Panel shifted to the left side
    fig.add_annotation(
        text=top_3_text,
        xref="paper", yref="paper",
        x=0.02, y=0.95,       # CHANGED: x shifted from 0.98 to 0.02 to mount it on the left edge
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Arial, sans-serif", size=11, color="#333333")
    )

    fig.update_layout(
        title=f"Ascending Activations for Layer {layer}, Neuron {neuron_idx}",
        xaxis_title="Prompts Count (Ordered by Activation)",
        yaxis_title="Activation Value",
        xaxis=dict(
            type="linear",
            showticklabels=True,
            showgrid=False,
            nticks=10 # Distributes clean numerical labels at regular step intervals
        ),
        template="plotly_white",
        clickmode='event+select'
    )

    raw_html = fig.to_html(include_plotlyjs='cdn')
    
    # Custom DOM layout script handling interactive click logic adjustments
    custom_js = """
    <script>
    window.addEventListener('DOMContentLoaded', (event) => {
        var plotDiv = document.getElementsByClassName('plotly-graph-div');
        if (plotDiv) {
            var activeAnnotations = {};
            plotDiv.on('plotly_click', function(data){
                if(data.points.length > 0){
                    var point = data.points;
                    var pointId = point.x;
                    
                    // Toggle annotation off if already selected
                    if (activeAnnotations[pointId]) {
                        var remaining = Object.values(activeAnnotations).filter(a => a.id !== pointId);
                        activeAnnotations = {};
                        remaining.forEach(a => { activeAnnotations[a.id] = a; });
                        Plotly.relayout(plotDiv, { 'annotations': remaining });
                    } else {
                        // Extract target metadata array variables passed downstream
                        var fullPrompt = point.customdata;
                        var newAnnotation = {
                            id: pointId,
                            x: point.x,
                            y: point.y,
                            text: "<b>Locked Prompt:</b><br>" + fullPrompt,
                            showarrow: true,
                            arrowhead: 2,
                            ax: 0,
                            ay: -100,
                            bgcolor: 'rgba(255, 255, 255, 0.95)',
                            bordercolor: point.fullData.marker.color[point.pointNumber] || '#333',
                            borderwidth: 2,
                            borderpad: 4,
                            font: {size: 11}
                        };
                        activeAnnotations[pointId] = newAnnotation;
                        Plotly.relayout(plotDiv, { 'annotations': Object.values(activeAnnotations) });
                    }
                }
            });
        }
    });
    </script>
    """
    
    final_html = raw_html.replace("</body>", custom_js + "</body>")
    
    output_dir = "neuron_explorer_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"neuron_L{layer}_N{neuron_idx}.html")

    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write(final_html)
        
    print(f"Interactive dashboard file generated at: {output_filename}")

if __name__ == "__main__":
    # Define parameters here
    CHOSEN_LAYER = 5
    CHOSEN_NEURON = 1368
    
    create_interactive_chart(layer=CHOSEN_LAYER, neuron_idx=CHOSEN_NEURON)
