from continued_pretraining import run_continued_pretraining

# from convert_composer_to_hf import convert_composer_to_hf
# from evaluate_model import evaluate_model
from pipeline import PipelineConfig, register_exit_handler

if __name__ == "__main__":
    register_exit_handler("1.3b to 370m")

    config = PipelineConfig(
        from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
        from_model_size="1.3b",
        to_model_size="370m",
    )

    run_continued_pretraining(config.continued_pretraining_config)

    # Step 6: Evaluate HF original from_model
    # print("Step 6: Evaluating original HF model")
    # evaluate_model(config.from_model_eval_config)

    # Step 6.5: Convert to HF
    # print("Step 6.5: Converting pruned model to HF")
    # convert_composer_to_hf(config.pruned_to_hf_config)

    # print("Step 6.5: Converting continued pretraining model to HF")
    # convert_composer_to_hf(config.continued_pretraining_to_hf_config)

    # # Step 7: Evaluate HF pruning result model
    # print("Step 7: Evaluating pruning result model")
    # evaluate_model(config.pruning_eval_config)

    # # Step 8: Evaluate HF continued pretraining result model
    # print("Step 8: Evaluating continued pretraining result model")
    # evaluate_model(config.continued_pretraining_eval_config)
