import os
import subprocess
from dataclasses import dataclass

from data_types import job_memory
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConvertHfToComposerConfig:
    project_root: str  # eg. "/nvmefs1/daranhe/llm-shearing"
    hf_model_name: str  # eg. "princeton-nlp/Sheared-LLaMA-1.3B"
    model_size: str  # eg. "1.3b"
    output_path: str = None  # type: ignore

    def __post_init__(self):
        if self.model_size and self.model_size[0].isnumeric():
            self.model_size = self.model_size.upper()
        if self.output_path is None:
            self.output_path = f"{self.project_root}/models/Sheared-LLaMA-{self.model_size}-composer/state_dict.pt"


@job_memory.cache
def convert_hf_to_composer(config: ConvertHfToComposerConfig) -> None:
    # Create the necessary directory if it doesn't exist
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    # Convert the Hugging Face model to Composer key format
    command = [
        "python3",
        "-m",
        "llmshearing.utils.composer_to_hf",
        "save_hf_to_composer",
        config.hf_model_name,
        config.output_path,
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)

    # Test the conversion
    command = [
        "python3",
        "-m",
        "llmshearing.utils.test_composer_hf_eq",
        config.hf_model_name,
        config.output_path,
        config.model_size,
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
