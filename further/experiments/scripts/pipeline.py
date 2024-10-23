import atexit
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict

import requests
from continued_pretraining import ContinuedPretrainingConfig, run_continued_pretraining
from convert_composer_to_hf import ConvertComposerToHfConfig, convert_composer_to_hf
from convert_hf_to_composer import ConvertHfToComposerConfig, convert_hf_to_composer
from convert_to_pruned_model import ConvertToPrunedModelConfig, prune_and_convert_model
from dataclasses_json import dataclass_json
from evaluate_model import EvaluationConfig, evaluate_model
from pruning import NetworkDims, PruningConfig, run_pruning


@dataclass_json
@dataclass
class PipelineConfig:
    from_model_name: str  # HF name or path
    from_model_size: str
    to_model_size: str

    project_root: str = "/nvmefs1/daranhe/llm-shearing"
    data_dir: str = "/nvmefs1/daranhe/llm-shearing/data/for_prune"
    project_output_dir: str | None = None
    to_model_dims: NetworkDims | None = None

    hf_to_composer_config: ConvertHfToComposerConfig = field(init=False)
    pruning_config: PruningConfig = field(init=False)
    convert_to_pruned_config: ConvertToPrunedModelConfig = field(init=False)
    continued_pretraining_config: ContinuedPretrainingConfig = field(init=False)

    pruned_to_hf_config: ConvertComposerToHfConfig = field(init=False)
    continued_pretraining_to_hf_config: ConvertComposerToHfConfig = field(init=False)
    from_model_eval_config: EvaluationConfig = field(init=False)
    pruning_eval_config: EvaluationConfig = field(init=False)
    continued_pretraining_eval_config: EvaluationConfig = field(init=False)

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
                raise ValueError(f"Unsupported to_model_size: {self.to_model_size}")

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
            data_dir="/nvmefs1/daranhe/llm-shearing/out/data_preparation/mds/for_ft",
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

        self.from_model_eval_config = EvaluationConfig(
            model=self.from_model_name,
            **self.get_overrides("from_model_eval"),
        )

        self.pruning_eval_config = EvaluationConfig(
            model=self.pruned_to_hf_config.output_path,
            **self.get_overrides("pruning_eval"),
        )

        self.continued_pretraining_eval_config = EvaluationConfig(
            model=self.pruned_to_hf_config.output_path,
            **self.get_overrides("continued_pretraining_eval"),
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

    # Step 2.5: Convert to pruned model
    print("Step 2.5: Converting to pruned model")
    prune_and_convert_model(config.convert_to_pruned_config)

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


def send_pushover_notification(user_key, api_token, message, title="MLDS Job Status"):
    """
    Send a Pushover notification.

    :param user_key: Your Pushover user key
    :param api_token: Your Pushover application's API token
    :param message: The message to send
    :param title: The title of the notification (default: "Job Status")
    :return: True if the notification was sent successfully, False otherwise
    """
    url = "https://api.pushover.net/1/messages.json"
    data = {"token": api_token, "user": user_key, "title": title, "message": message}

    response = requests.post(url, data=data)

    if response.status_code == 200:
        print("Notification sent successfully")
        return True
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")
        return False


def exit_handler(user_key, api_token, job_name):
    exit_code = sys.exc_info()[1]
    if exit_code is None:
        status = "completed successfully"
    else:
        status = f"failed with exit code {exit_code}"

    message = f"Your job {job_name} has {status}."
    send_pushover_notification(user_key, api_token, message)


def register_exit_handler(job_name: str):
    """
    Register the exit handler to send a notification when the script exits.
    """

    user_key, api_token = os.getenv("PUSHOVER_USER_KEY"), os.getenv(
        "PUSHOVER_API_TOKEN"
    )

    assert user_key, "PUSHOVER_USER_KEY is not set"
    assert api_token, "PUSHOVER_API_TOKEN is not set"

    send_pushover_notification(user_key, api_token, f"Starting job {job_name}")

    atexit.register(exit_handler, user_key, api_token, job_name)
