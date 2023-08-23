#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import argparse
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field


@dataclass_json
@dataclass
class CoqBuildSpec:
    project_name: str
    train_files: typing.List[str] = field(default_factory=list)
    test_files: typing.List[str] = field(default_factory=list)
    build_partition: typing.Optional[str] = None
    build_command: typing.Optional[str] = None
    timeout : typing.Optional[int] = None
    switch: typing.Optional[str] = None

if __name__ == "__main__":
    # take the project split as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_split", type=str, required=False, default="data/proverbot9001/coqgym_projs_splits.json", help="path to the project split")
    args = parser.parse_args()
    with open(args.project_split, "r") as f:
        project_splits = CoqBuildSpec.schema().loads(f.read(), many=True)
        print(project_splits)