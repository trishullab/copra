import logging
import time
import os
from copra.agent.human_policy import HumanPolicy
from copra.agent.simple_proof_agent import ProofAgent
from copra.main.eval_benchmark import check_query_limit_reached, query_limit_info_message
from itp_interface.rl.simple_proof_env import ProofEnv, ProofExecutorCallback, ProofAction
from itp_interface.tools.log_utils import setup_logger

lemma_name = "imo_1959_p1"
file_path = "data/test/miniF2F-lean4/MiniF2F/Test.lean"
project_path = "data/test/miniF2F-lean4"
timeout_in_secs = 60
proof_exec_callback = ProofExecutorCallback(
        project_folder=project_path,
        file_path=file_path,
        language=ProofAction.Language.LEAN4,
        use_hammer=False ,
        timeout_in_secs=timeout_in_secs,
        use_human_readable_proof_context=True,
        suppress_error_log=True,
        always_use_retrieval=False,
        logger=None)
search_guidance_policy = HumanPolicy()
proof_dump_file_name = f"proof_dump_{lemma_name}.json"
time_str = time.strftime("%Y%m%d-%H%M%S")
log_dir = f".log/human_policy_run/{lemma_name}/{time_str}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "human_policy_run.log")
logger = setup_logger(
    __name__,
    log_file,
    logging.INFO,
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

with ProofEnv(f"basic_proof_env_{lemma_name}", proof_exec_callback, lemma_name) as env:
    agent = ProofAgent(f"proof_agent_{lemma_name}", search_guidance_policy, False, proof_dump_file_name, logger=logger)
    agent.run_episodes_till_stop(
        env,
        episodes=1,
        render=False,
        stop_policy=check_query_limit_reached(100),
        policy_info_message=query_limit_info_message(100))
