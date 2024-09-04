# Define the Hugging Face model name and the output path
PROJECT_ROOT=/nvmefs1/daranhe/llm-shearing
HF_MODEL_NAME=NousResearch/Llama-2-7b-hf
OUTPUT_PATH=${PROJECT_ROOT}/models/LLaMA2-7B-composer/state_dict.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH

MODEL_SIZE=7B
python3 -m llmshearing.utils.test_composer_hf_eq $HF_MODEL_NAME $OUTPUT_PATH $MODEL_SIZE