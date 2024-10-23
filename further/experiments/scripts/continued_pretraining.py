import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ContinuedPretrainingConfig:
    data_dir: str
    from_model: str
    to_model: str
    pruned_model_path: str

    proj_dir: str = "/nvmefs1/daranhe/llm-shearing"
    output_dir: str | None = None
    max_seq_len: int = 4096
    device_train_microbatch_size: int = 16
    global_train_batch_size: int = 256
    device_eval_batch_size: int = 8
    lr: float = 1e-4
    max_duration: str = "48000ba"
    save_interval: str = "1800ba"
    t_warmup: str = "1440ba"
    dynamic: bool = True
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
        default_factory=lambda: [0.2192, 0.0002, 0.0791, 0.0064, 0.0096, 0.001, 0.6845]
    )
    update_type: str = "doremi"
    target_loss: Optional[List[float]] = None
    eval_split_name: str = "eval_merge"
    eval_interval: str = "400ba"
    save_dir: str = None  # type: ignore

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = (
                f"{self.proj_dir}/out/pruning_{self.from_model}_to_{self.to_model}"
            )
        if self.save_dir is None:
            prune_run_name = f"llama2_{self.from_model}_pruning_scaling_doremi_to{self.to_model}_sl{self.max_seq_len}"
            run_name = f"{prune_run_name}_ft{self.max_duration}_continued_pretraining"
            self.save_dir = f"{self.output_dir}/{run_name}"

    def get_checkpoint_paths(self) -> List[Path]:
        pattern = "ep0-ba*-rank0.pt"
        return sorted(Path(self.save_dir).glob(pattern), key=self.get_checkpoint_id)

    @classmethod
    def get_checkpoint_id(cls, checkpoint_path: Union[Path, str]) -> int:
        return int(Path(checkpoint_path).name[len("ep0-ba") : -len("-rank0.pt")])


def to_list_str(lst: List[Any]):
    return f"[{','.join(map(str, lst))}]"


# @job_memory.cache
def run_continued_pretraining(config: ContinuedPretrainingConfig):
    train_script = f"{config.proj_dir}/LLM-Shearing/llmshearing/train.py"
    config_file = f"{config.proj_dir}/LLM-Shearing/llmshearing/configs/llama2/{config.to_model}.yaml"
    wandb_dir = config.save_dir

    # Set target loss based on to_model size if not provided
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

    assert config.target_loss is not None
    assert config.save_dir is not None

    cmd = [
        "python",
        "-m",
        "composer",
        train_script,
        config_file,
        f"run_name={config.save_dir.split('/')[-1]}",
        f"data_local={config.data_dir}",
        f"eval_loader.dataset.split={config.eval_split_name}",
        f"global_train_batch_size={config.global_train_batch_size}",
        f"device_train_microbatch_size={config.device_train_microbatch_size}",
        f"device_eval_batch_size={config.device_eval_batch_size}",
        f"max_seq_len={config.max_seq_len}",
        f"max_duration={config.max_duration}",
        "eval_first=true",
        f"scheduler.t_warmup={config.t_warmup}",
        f"save_folder={config.save_dir}",
        f"loggers.wandb.init_kwargs.dir={wandb_dir}",
        f"loggers.wandb.project=pruning-from-{config.from_model}",
        f"eval_interval={config.eval_interval}",
        f"save_interval={config.save_interval}",
        f"optimizer.lr={config.lr}",
        "model.l0_module=null",
        f"model.path={config.pruned_model_path}",
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

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
