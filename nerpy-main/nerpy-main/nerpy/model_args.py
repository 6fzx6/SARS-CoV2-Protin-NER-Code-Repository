# -*- coding: utf-8 -*-

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count

from torch.utils.data import Dataset


# ====================
# 工具函数
# ====================

def get_default_process_count():
    """
    获取默认进程数

    Returns:
        int: 默认进程数
    """
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)
    return process_count


def get_special_tokens():
    """
    获取特殊token列表

    Returns:
        list: 特殊token列表
    """
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


# ====================
# 模型参数类
# ====================

@dataclass
class ModelArgs:
    """
    模型基础参数类，包含训练和推理的各种配置参数
    """

    # Adafactor优化器参数
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True

    # Adam优化器参数
    adam_epsilon: float = 1e-8

    # 路径和目录设置
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"

    # 配置参数
    config: dict = field(default_factory=dict)

    # 调度器参数
    cosine_schedule_num_cycles: float = 0.5

    # 自定义参数设置
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)

    # 数据加载设置
    dataloader_num_workers: int = 0

    # 文本处理设置
    do_lower_case: bool = False

    # 量化设置
    dynamic_quantize: bool = False

    # 早停策略设置
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3

    # 编码设置
    encoding: str = None

    # 批处理大小
    eval_batch_size: int = 8
    train_batch_size: int = 8

    # 训练评估设置
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True

    # 混合精度训练
    fp16: bool = False

    # 梯度设置
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 学习率设置
    learning_rate: float = 4e-5
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0

    # 分布式训练
    local_rank: int = -1

    # 日志设置
    logging_steps: int = 50

    # 损失函数设置
    loss_type: str = None
    loss_args: dict = field(default_factory=dict)

    # 随机种子
    manual_seed: int = None

    # 序列长度
    max_seq_length: int = 128

    # 模型信息
    model_name: str = None
    model_type: str = None

    # 多进程设置
    multiprocessing_chunksize: int = -1
    process_count: int = field(default_factory=get_default_process_count)
    use_multiprocessing: bool = False
    use_multiprocessing_for_evaluation: bool = False

    # GPU设置
    n_gpu: int = 1

    # 缓存设置
    no_cache: bool = False
    reprocess_input_data: bool = True
    use_cached_eval_features: bool = False

    # 保存设置
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = True
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 2000

    # 训练设置
    num_train_epochs: int = 1

    # 优化器和调度器
    optimizer: str = "AdamW"
    scheduler: str = "linear_schedule_with_warmup"

    # 输出设置
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = False

    # 多项式衰减调度器参数
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0

    # 量化模型
    quantized_model: bool = False

    # 静默模式
    silent: bool = False

    # 特殊token处理
    skip_special_tokens: bool = True

    # TensorBoard设置
    tensorboard_dir: str = None

    # 线程设置
    thread_count: int = None

    # 分词器设置
    tokenizer_name: str = None
    tokenizer_type: str = None

    # 训练参数设置
    train_custom_parameters_only: bool = False

    # WandB设置
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None

    def update_from_dict(self, new_values):
        """
        从字典更新参数值

        Args:
            new_values (dict): 包含参数名和值的字典
        """
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise TypeError(f"{new_values} is not a Python dict.")

    def get_args_for_saving(self):
        """
        获取用于保存的参数字典

        Returns:
            dict: 可保存的参数字典
        """
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
            if key not in self.not_saved_args
        }

        # 清理wandb_kwargs中的特殊设置
        if "settings" in args_for_saving["wandb_kwargs"]:
            del args_for_saving["wandb_kwargs"]["settings"]

        # 移除dataset_class（不可序列化）
        if "dataset_class" in args_for_saving:
            del args_for_saving["dataset_class"]

        return args_for_saving

    def save(self, output_dir):
        """
        保存参数到指定目录

        Args:
            output_dir (str): 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()

            # 处理tokenizer_type的序列化
            if args_dict["tokenizer_type"] is not None and not isinstance(
                args_dict["tokenizer_type"], str
            ):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__

            json.dump(args_dict, f)

    def load(self, input_dir):
        """
        从指定目录加载参数

        Args:
            input_dir (str): 输入目录路径
        """
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)
                self.update_from_dict(model_args)


@dataclass
class NERArgs(ModelArgs):
    """
    NER模型参数类，继承自ModelArgs，添加NER特定参数
    """

    # 模型类名
    model_class: str = "NERModel"

    # 分类报告设置
    classification_report: bool = False

    # 数据集设置
    dataset_class: Dataset = None

    # 标签设置
    labels_list: list = field(default_factory=list)

    # 懒加载设置
    lazy_loading: bool = False
    lazy_loading_start_line: int = 0

    # ONNX设置
    onnx: bool = False

    # 特殊token列表
    special_tokens_list: list = field(default_factory=list)
