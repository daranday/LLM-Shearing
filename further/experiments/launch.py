import argparse
import subprocess as sp
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


class Runner:
    def __init__(self, args):
        self.args = args
        assert self.args.cpu or self.args.gpu, "Please specify --cpu or --gpu"

    def create_exp_args(self):
        slots_args = []
        if self.args.cpu:
            slots_args = ["--config", "resources.slots_per_trial=0"]
        elif self.args.gpu:
            slots_args = ["--config", f"resources.slots_per_trial={self.args.gpu}"]

        entrypoint_args = [
            "--config",
            f"entrypoint=bash python_loader.sh {' '.join(self.args.entrypoint)}",
        ]

        return [
            "det",
            "experiment",
            "create",
            *slots_args,
            *entrypoint_args,
            "--project_id",
            "253",
            "default_config.yaml",
            ".",
        ]

    def run(self):
        args = self.create_exp_args()
        print(f"Running with args: {args}")
        sp.run(args, cwd=SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Launching experiment")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--entrypoint", required=True, nargs="+", type=str)
    args = parser.parse_args()
    if isinstance(args.entrypoint, str):
        args.entrypoint = [args.entrypoint]
    return args


if __name__ == "__main__":
    args = parse_args()
    runner = Runner(args)
    runner.run()
