#!/usr/bin/env python3

import logging
import os
from copra.main.config import EvalProofResults
from dataclasses import dataclass
from dataclasses_json import dataclass_json

def diff_proof_results(a: EvalProofResults, b: EvalProofResults, logger: logging.Logger):
    logger.info(f"Comparing {a.path} and {b.path}")
    a_solved = {}
    b_solved = {}
    a_solved_b_solved = {}
    unsolved = {}
    for file in a.theorem_map.keys():
        if file not in b.theorem_map:
            for theorem in a.theorem_map[file].keys():
                if file not in a_solved:
                    a_solved[file] = []
                if a.theorem_map[file][theorem].proof_found:
                    a_solved[file].append(theorem)
                else:
                    if file not in unsolved:
                        unsolved[file] = []
                    if theorem not in unsolved[file]:
                        unsolved[file].append(theorem)
        else:
            for theorem in a.theorem_map[file].keys():
                if a.theorem_map[file][theorem].proof_found and b.theorem_map[file][theorem].proof_found:
                    if file not in a_solved_b_solved:
                        a_solved_b_solved[file] = []
                    a_solved_b_solved[file].append(theorem)
                elif b.theorem_map[file][theorem].proof_found:
                    if file not in b_solved:
                        b_solved[file] = []
                    b_solved[file].append(theorem)
                elif a.theorem_map[file][theorem].proof_found:
                    if file not in a_solved:
                        a_solved[file] = []
                    a_solved[file].append(theorem)
                else:
                    if file not in unsolved:
                        unsolved[file] = []
                    if theorem not in unsolved[file]:
                        unsolved[file].append(theorem)
    for file in b.theorem_map.keys():
        if file not in a.theorem_map:
            for theorem in b.theorem_map[file].keys():
                if b.theorem_map[file][theorem].proof_found:
                    if file not in b_solved:
                        b_solved[file] = []
                    b_solved[file].append(theorem)
                else:
                    if file not in unsolved:
                        unsolved[file] = []
                    if theorem not in unsolved[file]:
                        unsolved[file].append(theorem)
        else:
            for theorem in b.theorem_map[file].keys():
                if b.theorem_map[file][theorem].proof_found and \
                    (theorem not in a.theorem_map[file] or not a.theorem_map[file][theorem].proof_found):
                    if file not in b_solved:
                        b_solved[file] = []
                    if theorem not in b_solved[file]:
                        b_solved[file].append(theorem)
                elif not b.theorem_map[file][theorem].proof_found and \
                    (theorem not in a.theorem_map[file] or not a.theorem_map[file][theorem].proof_found):
                    if file not in unsolved:
                        unsolved[file] = []
                    if theorem not in unsolved[file]:
                        unsolved[file].append(theorem)
    logger.info(f"Number of theorems solved by only a: {sum([len(k) for k in a_solved.values()])}")
    logger.info(f"Number of theorems solved by only b: {sum([len(k) for k in b_solved.values()])}")
    logger.info(f"Number of theorems solved by both: {sum([len(k) for k in a_solved_b_solved.values()])}")
    logger.info(f"Number of unsolved theorems: {sum([len(k) for k in unsolved.values()])}")
    solved_dict ={
        a.path: a_solved,
        b.path: b_solved,
        f"both \n{a.path} \nand \n{b.path}": a_solved_b_solved,
        "unsolved": unsolved
    }
    logger.info(f"Displaying results \n {solved_dict}")
    display_text = ""
    for key in sorted(solved_dict.keys()):
        paths = list(solved_dict[key].keys())
        paths.sort()
        if key != "unsolved":
            display_text += f"""
# Solved by {key}
- files:
"""
        else:
            display_text += f"""
# Unsolved
- files:
"""
        for path in paths:
            theorems = solved_dict[key][path]
            theorems.sort()
            theorems_txt = '\n            - '.join(theorems)
            display_text += f"""
    - path: {path}
      theorems: 
        - {theorems_txt}
"""
    logger.info(f"\n\n{display_text}")

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, required=True)
    parser.add_argument("--b", type=str, required=True)
    parser.add_argument("--info", choices=["diff"], required=True)
    args = parser.parse_args()
    log_dir = ".log/stats/{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"stats_{args.info}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    with open(args.a, "r") as f:
        eval_proof_results_a = EvalProofResults.schema().loads(f.read())
    with open(args.b, "r") as f:
        eval_proof_results_b = EvalProofResults.schema().loads(f.read())
    if args.info == "diff":
        diff_proof_results(eval_proof_results_a, eval_proof_results_b, logger)
    else:
        raise NotImplementedError(f"Unknown info type: {args.info}")