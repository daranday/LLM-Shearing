model="${1:-princeton-nlp/Sheared-LLaMA-1.3B}"

python -m lm_eval --model hf \
    --model_args pretrained=$model \
    --tasks sciq,piqa,winogrande,arc_easy \
    --trust_remote_code \
    --device cuda:0 \
    --batch_size 8

# ,arc_challenge,hellaswag