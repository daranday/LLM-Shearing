import argparse
import base64
import subprocess
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

from continued_pretraining import ContinuedPretrainingConfig, run_continued_pretraining
from convert_composer_to_hf import ConvertComposerToHfConfig, convert_composer_to_hf
from convert_hf_to_composer import ConvertHfToComposerConfig, convert_hf_to_composer
from convert_to_pruned_model import ConvertToPrunedModelConfig, prune_and_convert_model
from dataclasses_json import dataclass_json
from evaluate_model import EvaluationConfig, evaluate_model
from pruning import PruningConfig, run_pruning

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass_json
@dataclass
class DummyConfig:
    a: int = 0
    b: str = "b"


def dummy(config: DummyConfig):
    print(config)


def launch_workflow(func: Callable, group: str, cpu: bool = False, gpu: int = 0):
    @wraps(func)
    def wrapper(*args):
        print(f"Launching workflow: {func.__name__}")
        launch_script = str(SCRIPT_DIR.parent / "launch.py")
        dispatch_script = str(SCRIPT_DIR / "dispatch.py")

        (config,) = args
        config_json_arg = base64.b64encode(config.to_json().encode()).decode()

        env_args = [f"--name={group} - {func.__name__}"]
        if cpu:
            env_args.append(f"--cpu")
        if gpu:
            env_args.append(f"--gpu={gpu}")
        subprocess.run(
            [
                "python",
                launch_script,
                *env_args,
                "--",
                "python",
                dispatch_script,
                func.__name__,
                config_json_arg,
            ]
        )

    return wrapper


def run_workflow(name: str, params_dict_arg_encoded: str):
    params_dict_arg = base64.b64decode(params_dict_arg_encoded).decode()
    if name == "convert_hf_to_composer":
        convert_hf_to_composer(ConvertHfToComposerConfig.from_json(params_dict_arg))
    elif name == "run_pruning":
        run_pruning(PruningConfig.from_json(params_dict_arg))
    elif name == "run_continued_pretraining":
        run_continued_pretraining(ContinuedPretrainingConfig.from_json(params_dict_arg))
    elif name == "convert_composer_to_hf":
        convert_composer_to_hf(ConvertComposerToHfConfig.from_json(params_dict_arg))
    elif name == "evaluate_model":
        evaluate_model(EvaluationConfig.from_json(params_dict_arg))
    elif name == "convert_to_pruned_model":
        prune_and_convert_model(ConvertToPrunedModelConfig.from_json(params_dict_arg))
    elif name == "dummy":
        dummy(DummyConfig.from_json(params_dict_arg))
    else:
        raise ValueError(f"Unsupported workflow name: {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a workflow step for the pipeline",
        usage="python pipeline.py <workflow_name> <params_dict_arg>",
    )
    parser.add_argument(
        "workflow_name", type=str, help="Name of the workflow step to run"
    )
    parser.add_argument(
        "params_dict_arg",
        type=str,
        help="JSON string of the parameters for the workflow step",
    )
    args = parser.parse_args()

    run_workflow(args.workflow_name, args.params_dict_arg)
