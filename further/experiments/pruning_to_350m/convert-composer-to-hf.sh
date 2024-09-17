MODEL_DIR=/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m/llama2_1.3b-sheared_pruning_scaling_doremi_to350m_sl4096
MODEL_PATH=$MODEL_DIR/pruned-ep0-ba3200-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-pruned-350m
MODEL_CLASS=LlamaForCausalLM
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
NUM_HIDDEN_LAYERS=20
INTERMEDIATE_SIZE=4096
MODEL_NAME=sheared-350m

echo python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}
