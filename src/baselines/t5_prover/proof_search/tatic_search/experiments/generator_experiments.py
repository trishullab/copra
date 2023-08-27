import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import hydra
from src.baselines.t5_prover.llm_helpers.experiment_config import ExperimentConfig, TheoremProvingExperiment, code_t5_small_trainable_factory, code_t5_base_trainable_factory, code_t5_large_trainable_factory
from src.baselines.t5_prover.llm_helpers.data_format_layout import DataFormatLayoutTypes
from src.baselines.t5_prover.llm_helpers.trainable_model import TrainingDataArguments, PaddingPolicy
from transformers import TrainingArguments, SchedulerType
from datetime import datetime


trainable_model_factory = {
    'code_t5_base_trainable_factory': code_t5_base_trainable_factory,
    'code_t5_small_trainable_factory': code_t5_small_trainable_factory,
    'code_t5_large_trainable_factory': code_t5_large_trainable_factory
}

def get_data_format_type(data_format_type: str):
    if data_format_type.startswith('DataFormatLayoutTypes'):
        data_format_type = data_format_type.split('.')[-1]
    return DataFormatLayoutTypes[data_format_type]

def get_scheduler_type(scheduler_type: str):
    if scheduler_type.startswith('SchedulerType'):
        scheduler_type = scheduler_type.split('.')[-1]
    return SchedulerType[scheduler_type]

def get_padding_policy(padding_policy: str):
    if padding_policy.startswith('PaddingPolicy'):
        padding_policy = padding_policy.split('.')[-1]
    return PaddingPolicy[padding_policy]

def set_cuda_visible_devices(cuda_visible_devices: list):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cuda_visible_devices])

@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    os.chdir(root_dir)
    print(cfg)
    if cfg.cuda_visible_devices:
        set_cuda_visible_devices(cfg.cuda_visible_devices)
    from torch.multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = cfg.model.experiment_config.experiment_name + "_" + time
    model_out_dir = os.path.join(cfg.model.model_out_base_dir, experiment_name)
    model_log_dir = os.path.join(cfg.model.training_args.logging_dir, time)
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_log_dir, exist_ok=True)
    experiment_setups = [TheoremProvingExperiment(
        enabled=cfg.model.enabled,
        trainable_model_factory = trainable_model_factory[cfg.model.trainable_model_factory],
        experiment_config = ExperimentConfig(
            experiment_name = experiment_name,
            model_name= cfg.model.experiment_config.model_name,
            data_format_type = get_data_format_type(cfg.model.experiment_config.data_format_type),
            commet_secret_name = cfg.model.experiment_config.commet_secret_name,
            training_folder_name = cfg.model.experiment_config.training_folder_name,
            test_folder_name = cfg.model.experiment_config.test_folder_name,
            top_k = cfg.model.experiment_config.top_k,
            model_param_top_p= cfg.model.experiment_config.model_param_top_p,
            model_param_top_k= cfg.model.experiment_config.model_param_top_k,
            model_param_temperature= cfg.model.experiment_config.model_param_temperature,
            test_sample_percentage = cfg.model.experiment_config.test_sample_percentage,
            num_gen_examples_dumps = cfg.model.experiment_config.num_gen_examples_dumps,
            max_len_x = cfg.model.training_data_args.max_length_x,
            max_len_y = cfg.model.training_data_args.max_length_y,
            max_sub_part_lens = [(length, trim_from_end) for length, trim_from_end in zip(cfg.model.experiment_config.max_sub_part_lens1, cfg.model.experiment_config.max_sub_part_lens2)], # 300 + 300 + 150 = 750, 150 for proof step
            tokenize_during_loss_computation = cfg.model.experiment_config.tokenize_during_loss_computation,
            use_data_parallel = cfg.model.experiment_config.use_data_parallel,
            train_percentage = cfg.model.experiment_config.train_percentage,
            cuda_visible_devices = cfg.cuda_visible_devices,
            training_data_args = TrainingDataArguments(
                                padding=cfg.model.training_data_args.padding, 
                                truncation=cfg.model.training_data_args.truncation, 
                                max_len_x=cfg.model.training_data_args.max_length_x,
                                max_len_y=cfg.model.training_data_args.max_length_y,
                                padding_policy=get_padding_policy(cfg.model.training_data_args.padding_policy), 
                                ignore_input_longer_than_block_size=cfg.model.training_data_args.ignore_input_longer_than_block_size, 
                                shuffle=cfg.model.training_data_args.shuffle),
            training_args = TrainingArguments(
                            output_dir=model_out_dir,
                            do_train=cfg.model.training_args.do_train, 
                            do_eval=cfg.model.training_args.do_eval,
                            per_device_train_batch_size=cfg.model.training_args.per_device_train_batch_size,
                            per_device_eval_batch_size=cfg.model.training_args.per_device_eval_batch_size,
                            seed=cfg.model.training_args.seed,
                            overwrite_output_dir=cfg.model.training_args.overwrite_output_dir, 
                            logging_dir=model_log_dir,
                            num_train_epochs=cfg.model.training_args.num_train_epochs,
                            eval_steps=cfg.model.training_args.eval_steps,
                            save_steps=cfg.model.training_args.save_steps,
                            logging_steps=cfg.model.training_args.logging_steps,
                            save_total_limit=cfg.model.training_args.save_total_limit,
                            evaluation_strategy=cfg.model.training_args.evaluation_strategy,
                            learning_rate=cfg.model.training_args.learning_rate,
                            adam_beta1=cfg.model.training_args.adam_beta1,
                            adam_beta2=cfg.model.training_args.adam_beta2,
                            warmup_ratio=cfg.model.training_args.warmup_ratio,
                            weight_decay=cfg.model.training_args.weight_decay,
                            lr_scheduler_type=get_scheduler_type(cfg.model.training_args.lr_scheduler_type),
                            load_best_model_at_end=cfg.model.training_args.load_best_model_at_end,
                            metric_for_best_model=cfg.model.training_args.metric_for_best_model,
                            greater_is_better=cfg.model.training_args.greater_is_better))
    )]

    for experiment in experiment_setups:
        experiment.run()
    pass

if __name__ == "__main__":
    main()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # result = []
    # with Pool() as pool:
    #     # for experiment in experiment_setups:
    #     #     result.append(pool.apply_async(experiment.run))

    #     for r in result:
    #         r.wait()

# CUDA_VISIBLE_DEVICES="1,3" nohup python3 src/proof_search/tatic_search/experiments/code_t5_generator_experiments.py &