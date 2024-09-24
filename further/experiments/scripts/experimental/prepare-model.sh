# Define the Hugging Face model name and the output path
# PROJECT_ROOT=/nvmefs1/daranhe/llm-shearing
# HF_MODEL_NAME=princeton-nlp/Sheared-LLaMA-1.3B
# OUTPUT_PATH=${PROJECT_ROOT}/models/Sheared-LLaMA-1.3B-composer/state_dict.pt
# MODEL_SIZE=1.3B

set -x

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH

python3 -m llmshearing.utils.test_composer_hf_eq $HF_MODEL_NAME $OUTPUT_PATH $MODEL_SIZE