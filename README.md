# Sigmund
Sigmund is an ongoing research project exploring Mechanistic Interpretability using lightweight models and specialized toy problems. The goal is to reverse-engineer how language models process specific concepts by mapping internal neural activations. You can find my preliminary experiments on Medium. 

## prompts.txt
A list of prompts separated by concepts (or tags).


## SmolLM2–135M.py
Reads the prompts from `prompts.txt` and runs them through the chosen model (defined in X). 

For each neuron, it keeps the peak activation across all tokens in the prompt. It outputs a .npz with the following info for each prompt:

`
    np.savez_compressed(
        npz_filename,
        neurons=aggregated_neurons, 
        attentions=aggregated_heads,
        prompt=PROMPT,
        tag=TAG,
        tag_description=TAG_DESC
    )
`
It also outputs `neuron_peak_tokens.txt` to keep track of the tokens that caused higher activations.

## prompt_heatmap_visualizer.py
Interactive heatmap visualization of the top neuron activations for each prompt. It normalizes the data via quartile min-max. It produces an `html`file to the folder `Prompts_Heatmap`.

## prompt_heatmap_Consensus.py
Interactive heatmap visualization of the top neuron activations grouped by concept. `consensus_threshold` defines the percentage of prompts in the concept that have a given neuron marked as a top one (i.e., set to 1 after normalization). `leakage_threshold` controls how exclusively a neuron fires for a specific concept. Setting this to 5% means the neuron cannot be highly active in more than 5% of prompts for any other concept

This results in three states  for each neuron in each concept — inactive (0), shared consensus (1.0 - RED), specialized (2.0 - white). It outputs an `html` dashboard to the folder `Concept_Heatmaps`.

## Neuron_explorer.py
Interactive neuron inspection. Displays all activations for a specific neuron selected through the variables `CHOSEN_LAYER`and `CHOSEN_NEURON`. 









