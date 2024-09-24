import subprocess
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    model: str = "princeton-nlp/Sheared-LLaMA-1.3B"
    tasks: List[str] = field(
        default_factory=lambda: ["sciq", "piqa", "winogrande", "arc_easy"]
    )
    device: str = "cuda:0"
    batch_size: int = 8
    trust_remote_code: bool = True


def evaluate_model(config: EvaluationConfig):
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
    ]

    if config.trust_remote_code:
        command.append("--trust_remote_code")

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
