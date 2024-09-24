import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConvertToPrunedModelConfig:
    model_path: str
    output_path: str | None = None

    def __post_init__(self):
        assert self.model_path.endswith(".pt")
        if self.output_path is None:
            p = Path(self.model_path)
            self.output_path = str(p.parent / f"pruned-{p.name}")


def prune_and_convert_model(config: ConvertToPrunedModelConfig):
    command = [
        "python3",
        "-m",
        "llmshearing.utils.post_pruning_processing",
        "prune_and_save_model",
        config.model_path,
    ]

    subprocess.run(command, check=True)

    assert Path(config.output_path).exists()


# Example usage
if __name__ == "__main__":
    MODEL_PATH = "/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m/llama2_1.3b-sheared_pruning_scaling_doremi_to350m_sl4096/latest-rank0.pt"
    prune_and_convert_model(MODEL_PATH)
