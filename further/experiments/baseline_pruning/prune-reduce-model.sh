MODEL_PATH=/nvmefs1/daranhe/llm-shearing/out/pruning_7b_to_1.3b/llama2_7b_pruning_scaling_doremi_to1.3b_sl4096/latest-rank0.pt
python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model $MODEL_PATH
