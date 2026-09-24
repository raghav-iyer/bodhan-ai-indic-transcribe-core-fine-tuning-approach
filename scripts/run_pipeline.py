#!/usr/bin/env python3
"""Run the complete validation-gated comparison with one command."""

import argparse

from marathi_asr.common import load_config, load_local_token
from marathi_asr.experiments import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/marathi.json")
    args = parser.parse_args()
    load_local_token()
    run_experiment(load_config(args.config))


if __name__ == "__main__":
    main()
