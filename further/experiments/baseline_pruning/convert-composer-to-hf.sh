MODEL_DIR=/nvmefs1/daranhe/llm-shearing/out/pruning_7b_to_1.3b/llama2_7b_pruning_scaling_doremi_to1.3b_sl4096/
MODEL_PATH=$MODEL_DIR/pruned-latest-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-latest_rank0
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=24
INTERMEDIATE_SIZE=5504
MODEL_NAME=Sheared-Llama-1.3B

python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}
