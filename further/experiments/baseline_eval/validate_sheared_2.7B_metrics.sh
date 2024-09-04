python -m lm_eval --model hf \
    --model_args pretrained=princeton-nlp/Sheared-LLaMA-2.7B \
    --tasks sciq,piqa,winogrande,arc_easy,arc_challenge \
    --trust_remote_code \
    --device cuda:0 \
    --batch_size 8
