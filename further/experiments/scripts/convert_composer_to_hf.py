import os
import subprocess
from dataclasses import dataclass

from data_types import NetworkDims


@dataclass
class ConvertComposerToHfConfig:
    model_path: str
    model_name: str
    output_path: str | None = None
    model_class: str = "LlamaForCausalLM"
    network_dims: NetworkDims = NetworkDims()

    def __post_init__(self):
        assert self.model_path.endswith(".pt")
        model_dir = os.path.dirname(self.model_path)
        if self.output_path is None:
            self.output_path = f"{model_dir}/hf-{self.model_name}"


def convert_composer_to_hf(config: ConvertComposerToHfConfig):

    command = [
        "python3",
        "-m",
        "llmshearing.utils.composer_to_hf",
        "save_composer_to_hf",
        config.model_path,
        config.output_path,
        f"model_class={config.model_class}",
        f"hidden_size={config.network_dims.hidden_size}",
        f"num_attention_heads={config.network_dims.num_attention_heads}",
        f"num_hidden_layers={config.network_dims.num_hidden_layers}",
        f"intermediate_size={config.network_dims.intermediate_size}",
        f"num_key_value_heads={config.network_dims.num_attention_heads}",
        f"_name_or_path={config.model_name}",
    ]

    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":

    # Example usage:
    convert_composer_to_hf(
        ConvertComposerToHfConfig(
            model_dir="/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m/llama2_1.3b-sheared_pruning_scaling_doremi_to350m_sl4096",
            output_path="/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m/llama2_1.3b-sheared_pruning_scaling_doremi_to350m_sl4096/hf-pruned-350m",
            network_dims=NetworkDims(
                hidden_size=1024,
                num_attention_heads=16,
                num_hidden_layers=20,
                intermediate_size=4096,
            ),
        )
    )
