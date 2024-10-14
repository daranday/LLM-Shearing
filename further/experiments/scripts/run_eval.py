from pathlib import Path

from convert_composer_to_hf import ConvertComposerToHfConfig
from evaluate_model import EvaluationConfig, evaluate_model
from pipeline import PipelineConfig

# from continued_pretraining import run_continued_pretraining
from further.experiments.scripts.dispatch import launch_workflow

if __name__ == "__main__":

    configs = {
        "7b_to_370m": PipelineConfig(
            from_model_name="NousResearch/Llama-2-7b-hf",
            from_model_size="7b",
            to_model_size="370m",
        ),
        "1.3b_to_370m": PipelineConfig(
            from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
            from_model_size="1.3b",
            to_model_size="370m",
        ),
    }

    # # Step 6.5: Convert to HF
    for exp_name, config in configs.items():
        assert config.to_model_dims is not None

        pretrain_dir = Path(config.continued_pretraining_config.save_dir)
        pattern = "ep0-ba*-rank0.pt"
        checkpoint_paths = list(pretrain_dir.glob(pattern))
        
        for checkpoint_path in checkpoint_paths:
            ba = int(checkpoint_path.name[len("ep0-ba") : -len("-rank0.pt")])

            export_config = ConvertComposerToHfConfig(
                model_path=f"{config.continued_pretraining_config.save_dir}/ep0-ba{ba}-rank0.pt",
                model_name=f"hf-continued-pretrained-{config.to_model_size}-ba{ba}",
                network_dims=config.to_model_dims,
            )

            eval_config = EvaluationConfig(
                model=export_config.output_path,
            )

            # print("Step: Exporting continued pretraining model to HF")
            # launch_workflow(
            #     convert_composer_to_hf, group=f"{exp_name}-ba{ba}", gpu=2, detached=True
            # )(export_config)

            print("Step: Evaluating continued pretraining model")
            launch_workflow(
                evaluate_model, group=f"{exp_name}-ba{ba}", gpu=2, detached=True
            )(eval_config)
