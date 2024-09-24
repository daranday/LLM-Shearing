from continued_pretraining import run_continued_pretraining
from convert_composer_to_hf import convert_composer_to_hf
from convert_hf_to_composer import convert_hf_to_composer
from evaluate_model import evaluate_model
from pipeline import PipelineConfig
from pruning import run_pruning
from dispatch import launch_workflow

if __name__ == "__main__":

    config = PipelineConfig(
        from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
        from_model_size="1.3b",
        to_model_size="370m",
    )

    # Step 1: Convert from_model HF to Composer
    print("Step 1: Converting HF model to Composer format")
    launch_workflow(convert_hf_to_composer, cpu=True)(config.hf_to_composer_config)

    # # Step 2: Run pruning
    # print("Step 2: Running pruning")
    # run_pruning(config.pruning_config)

    # # Step 3: Run continued pretraining
    # print("Step 3: Running continued pretraining")
    # run_continued_pretraining(config.continued_pretraining_config)

    # # Step 4: Convert pruning result Composer model to HF
    # print("Step 4: Converting pruning result to HF format")
    # convert_composer_to_hf(config.pruned_to_hf_config)

    # # Step 5: Convert continued pretraining result Composer model to HF
    # print("Step 5: Converting continued pretraining result to HF format")
    # convert_composer_to_hf(config.continued_pretraining_to_hf_config)

    # # Step 6: Evaluate HF original from_model
    # print("Step 6: Evaluating original HF model")
    # evaluate_model(config.from_model_eval_config)

    # # Step 7: Evaluate HF pruning result model
    # print("Step 7: Evaluating pruning result model")
    # evaluate_model(config.pruning_eval_config)

    # # Step 8: Evaluate HF continued pretraining result model
    # print("Step 8: Evaluating continued pretraining result model")
    # evaluate_model(config.continued_pretraining_eval_config)
