import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from data_types import job_memory
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ConvertHfToComposerConfig:
    project_root: str  # eg. "/nvmefs1/daranhe/llm-shearing"
    hf_model_name: str  # eg. "princeton-nlp/Sheared-LLaMA-1.3B"
    model_size: str  # eg. "1.3b"
    output_path: Optional[str] = None

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to Composer format"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/nvmefs1/daranhe/llm-shearing",
        help="Root directory of the project",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="princeton-nlp/Sheared-LLaMA-1.3B",
        required=True,
        help="Name of the Hugging Face model to convert",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        default="1.3B",
        help="Size of the model",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the converted model"
    )

    args = parser.parse_args()

    config = ConvertHfToComposerConfig(
        project_root=args.project_root,
        hf_model_name=args.hf_model_name,
        model_size=args.model_size,
        output_path=args.output_path,
    )

    convert_hf_to_composer(config)
