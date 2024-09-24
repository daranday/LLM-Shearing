import argparse
from dataclasses import dataclass, field
from typing import Any, Dict

from continued_pretraining import ContinuedPretrainingConfig, run_continued_pretraining
from convert_composer_to_hf import ConvertComposerToHfConfig, convert_composer_to_hf
from convert_hf_to_composer import ConvertHfToComposerConfig, convert_hf_to_composer
from convert_to_pruned_model import ConvertToPrunedModelConfig
from evaluate_model import EvaluationConfig, evaluate_model
from pruning import NetworkDims, PruningConfig, run_pruning


@dataclass
class PipelineConfig:
    from_model_name: str  # HF name or path
    from_model_size: str
    to_model_size: str

    project_root: str = "/nvmefs1/daranhe/llm-shearing"
    data_dir: str = "/nvmefs1/daranhe/llm-shearing/data/for_prune"
    project_output_dir: str | None = None
    to_model_dims: NetworkDims | None = None

    hf_to_composer_config: ConvertHfToComposerConfig | None = None
    pruning_config: PruningConfig | None = None
    convert_to_pruned_config: ConvertToPrunedModelConfig | None = None
    continued_pretraining_config: ContinuedPretrainingConfig | None = None

    pruned_to_hf_config: ConvertComposerToHfConfig | None = None
    continued_pretraining_to_hf_config: ConvertComposerToHfConfig | None = None
    from_model_eval_config: EvaluationConfig | None = None
    pruning_eval_config: EvaluationConfig | None = None
    continued_pretraining_eval_config: EvaluationConfig | None = None

    overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.project_output_dir is None:
            self.project_output_dir = f"{self.project_root}/out/pruning_{self.from_model_size}_to_{self.to_model_size}"

        if self.to_model_dims is None:
            if self.to_model_size == "7b":
                self.to_model_dims = NetworkDims(
                    hidden_size=4096,
                    num_attention_heads=32,
                    num_hidden_layers=32,
                    intermediate_size=11008,
                )
            elif self.to_model_size == "13b":
                self.to_model_dims = NetworkDims(
                    hidden_size=5120,
                    num_attention_heads=40,
                    num_hidden_layers=40,
                    intermediate_size=13824,
                )
            elif self.to_model_size == "2.7b":
                self.to_model_dims = NetworkDims(
                    hidden_size=2560,
                    num_attention_heads=32,
                    num_hidden_layers=32,
                    intermediate_size=6912,
                )
            elif self.to_model_size == "1.3b":
                self.to_model_dims = NetworkDims(
                    hidden_size=2048,
                    num_attention_heads=16,
                    num_hidden_layers=24,
                    intermediate_size=5504,
                )
            elif self.to_model_size == "370m":
                self.to_model_dims = NetworkDims(
                    hidden_size=1024,
                    num_attention_heads=8,
                    num_hidden_layers=24,
                    intermediate_size=2816,
                )
            elif self.to_model_size == "350m":
                self.to_model_dims = NetworkDims(
                    hidden_size=1024,
                    num_attention_heads=16,
                    num_hidden_layers=20,
                    intermediate_size=4096,
                )
            else:
                raise ValueError(f"Unsupported to_model_size: {config.to_model_size}")

        self.hf_to_composer_config = ConvertHfToComposerConfig(
            project_root=self.project_root,
            hf_model_name=self.from_model_name,
            model_size=self.from_model_size,
            **self.get_overrides("hf_to_composer"),
        )

        self.pruning_config = PruningConfig(
            proj_dir=self.project_root,
            data_dir=self.data_dir,
            output_dir=self.project_output_dir,
            model_path=self.hf_to_composer_config.output_path,
            from_model=self.from_model_size,
            to_model=self.to_model_size,
            **self.get_overrides("pruning"),
        )

        self.convert_to_pruned_config = ConvertToPrunedModelConfig(
            model_path=f"{self.pruning_config.save_dir}/latest-rank0.pt",
            **self.get_overrides("convert_to_pruned"),
        )

        self.continued_pretraining_config = ContinuedPretrainingConfig(
            proj_dir=self.project_root,
            data_dir=self.data_dir,
            output_dir=self.project_output_dir,
            from_model=self.from_model_size,
            to_model=self.to_model_size,
            pruned_model_path=self.convert_to_pruned_config.output_path,
            **self.get_overrides("continued_pretraining"),
        )

        self.pruned_to_hf_config = ConvertComposerToHfConfig(
            model_path=f"{self.pruning_config.save_dir}/pruned-latest-rank0.pt",
            model_name=f"hf-pruned-{self.to_model_size}",
            network_dims=self.to_model_dims,
        )

        self.continued_pretraining_to_hf_config = ConvertComposerToHfConfig(
            model_path=f"{self.continued_pretraining_config.save_dir}/latest-rank0.pt",
            model_name=f"hf-continued-pretrained-{self.to_model_size}",
            network_dims=self.to_model_dims,
        )

    def get_overrides(self, prefix: str):
        return {
            k[len(prefix) + 1 :]: v
            for k, v in self.overrides.items()
            if k.startswith(prefix)
        }


def run_pipeline(config: PipelineConfig):
    # Step 1: Convert from_model HF to Composer
    print("Step 1: Converting HF model to Composer format")
    convert_hf_to_composer(config.hf_to_composer_config)

    # Step 2: Run pruning
    print("Step 2: Running pruning")
    run_pruning(config.pruning_config)

    # Step 3: Run continued pretraining
    print("Step 3: Running continued pretraining")
    run_continued_pretraining(config.continued_pretraining_config)

    # Step 4: Convert pruning result Composer model to HF
    print("Step 4: Converting pruning result to HF format")
    convert_composer_to_hf(config.pruned_to_hf_config)

    # Step 5: Convert continued pretraining result Composer model to HF
    print("Step 5: Converting continued pretraining result to HF format")
    convert_composer_to_hf(config.continued_pretraining_to_hf_config)

    # Step 6: Evaluate HF original from_model
    print("Step 6: Evaluating original HF model")
    evaluate_model(config.from_model_eval_config)

    # Step 7: Evaluate HF pruning result model
    print("Step 7: Evaluating pruning result model")
    evaluate_model(config.pruning_eval_config)

    # Step 8: Evaluate HF continued pretraining result model
    print("Step 8: Evaluating continued pretraining result model")
    evaluate_model(config.continued_pretraining_eval_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for model pruning and continued pretraining"
    )
    parser.add_argument(
        "--project_root", type=str, required=True, help="Root directory of the project"
    )
    parser.add_argument(
        "--from_model", type=str, required=True, help="Name of the source model"
    )
    parser.add_argument(
        "--from_model_size", type=str, required=True, help="Size of the source model"
    )
    parser.add_argument(
        "--to_model_size", type=str, required=True, help="Size of the target model"
    )
    parser.add_argument(
        "--hf_model_name", type=str, required=True, help="Hugging Face model name"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden size of the target model"
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        required=True,
        help="Number of attention heads of the target model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        required=True,
        help="Number of hidden layers of the target model",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=True,
        help="Intermediate size of the target model",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        project_root=args.project_root,
        from_model=args.from_model,
        from_model_size=args.from_model_size,
        to_model_size=args.to_model_size,
        to_model_network_dims=NetworkDims(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            intermediate_size=args.intermediate_size,
        ),
        hf_model_name=args.hf_model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    run_pipeline(config)
