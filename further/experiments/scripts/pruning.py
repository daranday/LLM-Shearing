import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

from data_types import NetworkDims, memory
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class PruningConfig:
    proj_dir: str
    data_dir: str
    model_path: str
    from_model: str
    to_model: str
    max_seq_len: int = 4096
    device_train_microbatch_size: int = 4
    global_train_batch_size: int = 32
    device_eval_batch_size: int = 8
    lr: float = 1e-4
    max_duration: str = "3200ba"
    save_interval: str = "3200ba"
    t_warmup: str = "320ba"
    dynamic: bool = True

    output_dir: str | None = None
    set_names: List[str] = field(
        default_factory=lambda: [
            "cc",
            "github",
            "book",
            "stackexchange",
            "wiki",
            "arxiv",
            "c4-rp",
        ]
    )
    proportion: List[float] = field(
        default_factory=lambda: [0.67, 0.045, 0.045, 0.02, 0.045, 0.025, 0.15]
    )
    update_type: str = "doremi"
    target_loss: Optional[List[float]] = None
    target_dims: Optional[NetworkDims] = None
    eval_split_name: str = "eval_merge"
    eval_target_model: bool = False
    eval_interval: str = "50ba"
    lag_lr: float = 1.0
    lagr_warmup: str = "640ba"

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = (
                f"{self.proj_dir}/out/pruning_{self.from_model}_to_{self.to_model}"
            )
        self.run_name = f"llama2_{self.from_model}_pruning_scaling_{self.update_type}_to{self.to_model}_sl{self.max_seq_len}"
        self.save_dir = f"{self.output_dir}/{self.run_name}"
        self.wandb_dir = self.save_dir


def to_list_str(lst: List[Any]):
    return f"[{','.join(map(str, lst))}]"


@memory.cache
def run_pruning(config: PruningConfig):
    # Set up paths
    train_script = f"{config.proj_dir}/LLM-Shearing/llmshearing/train.py"
    config_file = f"{config.proj_dir}/LLM-Shearing/llmshearing/configs/llama2/{config.from_model}.yaml"

    # Set target loss based on to_model if not provided
    if config.target_loss is None:
        if config.to_model == "1.3b":
            config.target_loss = [
                1.9643,
                0.7459,
                2.1393,
                1.6117,
                1.7590,
                1.4449,
                2.1251,
            ]
        elif config.to_model == "2.7b":
            config.target_loss = [
                1.8712,
                0.6883,
                2.0325,
                1.5353,
                1.6297,
                1.3560,
                2.0328,
            ]
        elif config.to_model in ["370m", "350m"]:
            config.target_loss = [2.1401, 0.8694, 2.3625, 1.7791, 2.047, 1.6637, 2.3139]

    if config.target_dims is None:
        if config.to_model == "7b":
            config.target_dims = NetworkDims(
                hidden_size=4096,
                num_attention_heads=32,
                num_hidden_layers=32,
                intermediate_size=11008,
            )
        elif config.to_model == "13b":
            config.target_dims = NetworkDims(
                hidden_size=5120,
                num_attention_heads=40,
                num_hidden_layers=40,
                intermediate_size=13824,
            )
        elif config.to_model == "2.7b":
            config.target_dims = NetworkDims(
                hidden_size=2560,
                num_attention_heads=32,
                num_hidden_layers=32,
                intermediate_size=6912,
            )
        elif config.to_model == "1.3b":
            config.target_dims = NetworkDims(
                hidden_size=2048,
                num_attention_heads=16,
                num_hidden_layers=24,
                intermediate_size=5504,
            )
        elif config.to_model == "370m":
            config.target_dims = NetworkDims(
                hidden_size=1024,
                num_attention_heads=8,
                num_hidden_layers=24,
                intermediate_size=2816,
            )
        elif config.to_model == "350m":
            config.target_dims = NetworkDims(
                hidden_size=1024,
                num_attention_heads=16,
                num_hidden_layers=20,
                intermediate_size=4096,
            )
        else:
            raise ValueError(f"Unsupported to_model: {config.to_model}")

    # Prepare command
    cmd = [
        "python",
        "-m",
        "composer",
        train_script,
        config_file,
        f"run_name={config.run_name}",
        f"data_local={config.data_dir}",
        f"eval_loader.dataset.split={config.eval_split_name}",
        f"global_train_batch_size={config.global_train_batch_size}",
        f"device_train_microbatch_size={config.device_train_microbatch_size}",
        f"device_eval_batch_size={config.device_eval_batch_size}",
        f"max_seq_len={config.max_seq_len}",
        f"max_duration={config.max_duration}",
        "eval_first=false",
        f"scheduler.t_warmup={config.t_warmup}",
        f"save_folder={config.save_dir}",
        f"loggers.wandb.init_kwargs.dir={config.wandb_dir}",
        f"eval_interval={config.eval_interval}",
        f"save_interval={config.save_interval}",
        f"optimizer.lr={config.lr}",
        f"optimizer.lag_lr={config.lag_lr}",
        f"model.path={config.model_path}",
        f"model.l0_module.lagrangian_warmup_steps={config.lagr_warmup}",
        "model.l0_module.pruning_modules=[head,intermediate,layer,hidden]",
        f"model.l0_module.eval_target_model={str(config.eval_target_model).lower()}",
        f"model.l0_module.target_model.d_model={config.target_dims.hidden_size}",
        f"model.l0_module.target_model.n_heads={config.target_dims.num_attention_heads}",
        f"model.l0_module.target_model.n_layers={config.target_dims.num_hidden_layers}",
        f"model.l0_module.target_model.intermediate_size={config.target_dims.intermediate_size}",
        "model.l0_module.target_model.vocab_size=32000",
        f"callbacks.data_loading.dynamic={str(config.dynamic)}",
        f"callbacks.data_loading.set_names={to_list_str(config.set_names)}",
        f"callbacks.data_loading.proportion={to_list_str(config.proportion)}",
        f"callbacks.data_loading.update_type={config.update_type}",
        f"callbacks.data_loading.target_loss={to_list_str(config.target_loss)}",
        "train_loader.num_workers=0",
        "train_loader.prefetch_factor=null",
        "train_loader.persistent_workers=false",
        "autoresume=false",
    ]

    # Run the command
    print(" ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)


# Example usage:
if __name__ == "__main__":
    run_pruning(
        PruningConfig(
            proj_dir="/nvmefs1/daranhe/llm-shearing",
            data_dir="/nvmefs1/daranhe/llm-shearing/data/for_prune",
            output_dir="/nvmefs1/daranhe/llm-shearing/out/pruning_from_1.3b_to_350m",
            model_path="/nvmefs1/daranhe/llm-shearing/models/Sheared-LLaMA-1.3B-composer/state_dict.pt",
            from_model="1.3b-sheared",
            to_model="350m",
        )
    )
