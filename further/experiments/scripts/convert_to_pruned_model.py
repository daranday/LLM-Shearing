import subprocess
from dataclasses import dataclass
from pathlib import Path

from data_types import job_memory
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConvertToPrunedModelConfig:
    model_path: str
    output_path: str = None  # type: ignore

    def __post_init__(self):
        assert self.model_path.endswith(".pt")
        if self.output_path is None:
            p = Path(self.model_path)
            self.output_path = str(p.parent / f"pruned-{p.name}")


@job_memory.cache
def prune_and_convert_model(config: ConvertToPrunedModelConfig):
    command = [
        "python3",
        "-m",
        "llmshearing.utils.post_pruning_processing",
        "prune_and_save_model",
        config.model_path,
    ]

    print(" ".join(command))
    subprocess.run(command, check=True)

    assert Path(config.output_path).exists()
