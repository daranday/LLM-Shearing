import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from data_types import job_memory
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EvaluationConfig:
    output_root: str = str("/nvmefs1/daranhe/llm-shearing/out/lm_eval")
    model: str = "princeton-nlp/Sheared-LLaMA-1.3B"
    tasks: List[str] = field(
        default_factory=lambda: ["sciq", "piqa", "winogrande", "arc_easy"]
    )
    device: str = "cuda:0"
    batch_size: int = 8
    trust_remote_code: bool = True
    output_path: str = None  # type: ignore

    def __post_init__(self):
        if self.output_path is None:
            model_path = Path(self.model)
            if model_path.is_absolute():
                self.output_path = str(
                    model_path.parent / f"{model_path.name}.lm_eval.json"
                )
            else:
                self.output_path = str(
                    Path(self.output_root) / f"{model_path}.lm_eval.json"
                )
                Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


@job_memory.cache
def evaluate_model(config: EvaluationConfig):
    # Hello
    python_bin: str = "/nvmefs1/daranhe/.conda/envs/evaluation/bin/python"

    command = [
        python_bin,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={config.model}",
        "--tasks",
        ",".join(config.tasks),
        "--device",
        config.device,
        "--batch_size",
        str(config.batch_size),
        "--output_path",
        str(config.output_root),
    ]

    if config.trust_remote_code:
        command.append("--trust_remote_code")

    print(" ".join(command))
    subprocess.run(command)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a language model.")
    parser.add_argument(
        "--model",
        type=str,
        default="princeton-nlp/Sheared-LLaMA-1.3B",
        help="The model to evaluate. Default is 'princeton-nlp/Sheared-LLaMA-1.3B'.",
    )

    args = parser.parse_args()

    config = EvaluationConfig(model=args.model)
    evaluate_model(config)

# Note: Uncomment the following line in the EvaluationConfig to add more tasks:
# tasks: List[str] = field(default_factory=lambda: ["sciq", "piqa", "winogrande", "arc_easy", "arc_challenge", "hellaswag"])
