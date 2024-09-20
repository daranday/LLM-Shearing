#!/usr/bin/env python3
import argparse
import subprocess as sp
from pathlib import Path

import yaml
from determined.common.experimental import Determined

SCRIPT_DIR = Path(__file__).resolve().parent


class Runner:
    def __init__(self, args):
        self.args = args
        assert self.args.cpu or self.args.gpu, "Please specify --cpu or --gpu"

    def get_config(self):
        config = yaml.safe_load(open(SCRIPT_DIR / "default_config.yaml"))

        config["entrypoint"] = (
            f"bash python_loader.sh {self.args.env} {' '.join(self.args.entrypoint)}"
        )

        if self.args.cpu:
            config["resources"]["slots_per_trial"] = 0
        elif self.args.gpu:
            config["resources"]["slots_per_trial"] = self.args.gpu

        if self.args.name is not None:
            config["name"] = self.args.name

        return config

    def run(self):
        client = Determined()

        exp = client.create_experiment(
            config=self.get_config(),
            model_dir=str(SCRIPT_DIR),
            project_id=253,
        )

        exp_url = f"{self.args.host}/det/experiments/{exp.id}/logs"

        print("Experiment started at {}".format(exp_url))

        try:
            sp.run(["det", "e", "logs", str(exp.id), "-f"], cwd=SCRIPT_DIR)
        except KeyboardInterrupt:
            to_kill_response = input("Kill experiment? [y/N]: ")
            to_kill = "y" in to_kill_response.lower()
            if to_kill:
                exp.kill()
                print(f"Experiment {exp.id} killed at {exp_url}")


def parse_args():
    parser = argparse.ArgumentParser(description="Launching experiment")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--env", default="default", choices=["default", "evaluation"])
    parser.add_argument(
        "entrypoint", nargs=argparse.REMAINDER, help="Positional arguments after --"
    )
    parser.add_argument(
        "--host", default="http://mlds-determined.us.rdlabs.hpecorp.net:8080"
    )
    args = parser.parse_args()
    assert "--" in args.entrypoint, "Please specify -- before entrypoint"
    args.entrypoint = args.entrypoint[args.entrypoint.index("--") + 1 :]
    assert args.entrypoint, "Please specify entrypoint"
    return args


if __name__ == "__main__":
    args = parse_args()
    runner = Runner(args)
    runner.run()
