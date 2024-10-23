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
from pipeline import PipelineConfig
from pruning import PruningConfig, run_pruning

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass_json
@dataclass
class DummyConfig:
    a: int = 0
    b: str = "b"


def dummy(config: DummyConfig):
    print(config)


def launch_workflow(
    func: Callable,
    group: str,
    cpu: bool = False,
    gpu: int = 0,
    detached: bool = False,
    no_follow: bool = False,
):
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
        if detached:
            env_args.append("--detached")
        if no_follow:
            env_args.append("--no-follow")
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
            ],
            check=True,
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


def run_pipeline_async(exp_name: str, config: PipelineConfig):

    # Step 1: Convert from_model HF to Composer
    # print("Step 1: Converting HF model to Composer format")
    # launch_workflow(convert_hf_to_composer, group=exp_name, gpu=2)(
    #     config.hf_to_composer_config
    # )

    # # Step 2: Run pruning
    # print("Step 2: Running pruning")
    # launch_workflow(run_pruning, group=exp_name, gpu=4)(config.pruning_config)

    # Step 2.5: Convert to pruned model
    print("Step 2.5: Convert to pruned model")
    launch_workflow(prune_and_convert_model, group=exp_name, gpu=2)(
        config.convert_to_pruned_config
    )

    # Step 3: Run continued pretraining
    print("Step 3: Running continued pretraining")
    launch_workflow(run_continued_pretraining, group=exp_name, gpu=4)(
        config.continued_pretraining_config
    )

    # Step 4: Convert pruning result Composer model to HF
    print("Step 4: Converting pruning result to HF format")
    launch_workflow(convert_composer_to_hf, group=exp_name, gpu=2)(
        config.pruned_to_hf_config
    )

    # Step 5: Convert continued pretraining result Composer model to HF
    print("Step 5: Converting continued pretraining result to HF format")
    launch_workflow(convert_composer_to_hf, group=exp_name, gpu=2)(
        config.continued_pretraining_to_hf_config
    )

    # Step 6: Evaluate HF original from_model
    print("Step 6: Evaluating original HF model")
    launch_workflow(evaluate_model, group=exp_name, gpu=4, detached=True)(
        config.from_model_eval_config
    )

    # Step 7: Evaluate HF pruning result model
    print("Step 7: Evaluating pruning result model")
    launch_workflow(evaluate_model, group=exp_name, gpu=4, detached=True)(
        config.pruning_eval_config
    )

    # Step 8: Evaluate HF continued pretraining result model
    print("Step 8: Evaluating continued pretraining result model")
    launch_workflow(evaluate_model, group=exp_name, gpu=4, detached=True)(
        config.continued_pretraining_eval_config
    )


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
