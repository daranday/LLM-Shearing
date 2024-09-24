MODEL_PATH=/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m/llama2_1.3b-sheared_pruning_scaling_doremi_to350m_sl4096/latest-rank0.pt
python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model $MODEL_PATH
