# -*- coding: utf-8 -*-

import collections
import math
import os
import random
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import (
    AlbertConfig,
    AlbertForTokenClassification,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    BertweetTokenizer,
    BigBirdConfig,
    BigBirdForTokenClassification,
    BigBirdTokenizer,
    DebertaConfig,
    DebertaForTokenClassification,
    DebertaTokenizer,
    DebertaV2Config,
    DebertaV2ForTokenClassification,
    DebertaV2Tokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForTokenClassification,
    ElectraTokenizer,
    HerbertTokenizerFast,
    LongformerConfig,
    LongformerForTokenClassification,
    LongformerTokenizer,
    MobileBertConfig,
    MobileBertForTokenClassification,
    MobileBertTokenizer,
    MPNetConfig,
    MPNetForTokenClassification,
    MPNetTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    SqueezeBertConfig,
    SqueezeBertForTokenClassification,
    SqueezeBertTokenizer,
    XLMConfig,
    XLMForTokenClassification,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForTokenClassification,
    XLNetTokenizerFast,
)
from transformers.convert_graph_to_onnx import convert, quantize
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

# 本地导入
from nerpy.bertspan import (
    BertSpanDataset,
    BertSpanForTokenClassification,
    SpanEntityScore,
    check_span_labels,
    get_span_subject,
)
from nerpy.losses import init_loss
from nerpy.model_args import NERArgs
from nerpy.ner_utils import (
    InputExample,
    LazyNERDataset,
    convert_examples_to_features,
    flatten_results,
    load_hf_dataset,
    read_examples_from_file,
)

# 尝试导入wandb
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

# 环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型配置常量
MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT = ["squeezebert", "deberta", "mpnet"]
MODELS_WITH_EXTRA_SEP_TOKEN = [
    "roberta",
    "xlmroberta",
    "longformer",
    "mpnet",
]
use_cuda = torch.cuda.is_available()


class NERModel:
    """
    命名实体识别(NER)模型类
    支持多种预训练模型如BERT, RoBERTa, DeBERTa等
    """

    def __init__(
        self,
        model_type,
        model_name,
        labels=None,
        weight=None,
        args=None,
        use_cuda=use_cuda,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
    ):
        """
        初始化NER模型

        Args:
            model_type: 模型类型 (bert, roberta等)
            model_name: 预训练模型名称或路径
            labels: 实体标签列表
            weight: 类别权重
            args: 模型参数
            use_cuda: 是否使用GPU
            cuda_device: 指定GPU设备
            onnx_execution_provider: ONNX执行提供者
        """
        self._setup_model_classes()
        self._initialize_args(model_name, args)
        self._set_random_seeds()
        self._configure_labels(model_type, model_name, labels)
        self._setup_device(use_cuda, cuda_device)
        self._initialize_model_and_tokenizer(
            model_type, model_name, weight, onnx_execution_provider, **kwargs
        )
        self._post_initialization_setup()

    # ====================
    # 初始化相关方法
    # ====================

    def _setup_model_classes(self):
        """设置模型类映射"""
        self.MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForTokenClassification, AlbertTokenizer),
            "auto": (AutoConfig, AutoModelForTokenClassification, AutoTokenizer),
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "bertspan": (BertConfig, BertSpanForTokenClassification, BertTokenizer),
            "bertweet": (
                RobertaConfig,
                RobertaForTokenClassification,
                BertweetTokenizer,
            ),
            "bigbird": (BigBirdConfig, BigBirdForTokenClassification, BigBirdTokenizer),
            "deberta": (DebertaConfig, DebertaForTokenClassification, DebertaTokenizer),
            "deberta-v2": (
                DebertaV2Config,
                DebertaV2ForTokenClassification,
                DebertaV2Tokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForTokenClassification,
                DistilBertTokenizer,
            ),
            "electra": (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
            "herbert": (BertConfig, BertForTokenClassification, HerbertTokenizerFast),
            "longformer": (
                LongformerConfig,
                LongformerForTokenClassification,
                LongformerTokenizer,
            ),
            "mobilebert": (
                MobileBertConfig,
                MobileBertForTokenClassification,
                MobileBertTokenizer,
            ),
            "mpnet": (MPNetConfig, MPNetForTokenClassification, MPNetTokenizer),
            "roberta": (
                RobertaConfig,
                RobertaForTokenClassification,
                RobertaTokenizerFast,
            ),
            "squeezebert": (
                SqueezeBertConfig,
                SqueezeBertForTokenClassification,
                SqueezeBertTokenizer,
            ),
            "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
            "xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForTokenClassification,
                XLMRobertaTokenizer,
            ),
            "xlnet": (XLNetConfig, XLNetForTokenClassification, XLNetTokenizerFast),
        }

    def _initialize_args(self, model_name, args):
        """初始化模型参数"""
        self.args = self._load_model_args(model_name)
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERArgs):
            self.args = args

    def _set_random_seeds(self):
        """设置随机种子"""
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

    def _configure_labels(self, model_type, model_name, labels):
        """配置标签列表"""
        if labels and self.args.labels_list:
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        elif "shibing624/bert4ner-base-chinese" == model_name:
            self.args.labels_list = [
                "I-ORG",
                "B-LOC",
                "O",
                "B-ORG",
                "I-LOC",
                "I-PER",
                "B-TIME",
                "I-TIME",
                "B-PER",
            ]
        elif "shibing624/bert4ner-base-uncased" == model_name:
            self.args.labels_list = [
                "E-ORG",
                "E-LOC",
                "S-MISC",
                "I-MISC",
                "S-PER",
                "E-PER",
                "B-MISC",
                "O",
                "S-LOC",
                "E-MISC",
                "B-ORG",
                "S-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
                "B-PER",
                "I-PER",
            ]
        elif "shibing624/bertspan4ner-base-chinese" == model_name:
            self.args.labels_list = ["O", "TIME", "PER", "LOC", "ORG"]
        else:
            self.args.labels_list = [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
            ]

        # 对于BERT Span模型，处理标签格式
        if model_type in ["bertspan"]:
            if self.args.labels_list and not check_span_labels(self.args.labels_list):
                self.args.labels_list = ["O"] + list(
                    set([i.split("-")[-1] for i in self.args.labels_list if i != "O"])
                )

        self.num_labels = len(self.args.labels_list)
        logger.debug(f"Using labels list: {self.args.labels_list}")

        # 处理大小写设置
        if "uncased" in model_name and not self.args.do_lower_case:
            self.args.do_lower_case = True
            logger.warning(f"Set do_lower_case=True for {model_name}")

    def _setup_device(self, use_cuda, cuda_device):
        """设置计算设备"""
        if not use_cuda:
            self.args.fp16 = False

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = "cpu"
        logger.debug(f"Device: {self.device}")

    def _initialize_model_and_tokenizer(
        self, model_type, model_name, weight, onnx_execution_provider, **kwargs
    ):
        """初始化模型和分词器"""
        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[model_type]

        # 配置模型
        if self.num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=self.num_labels, **self.args.config
            )
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        # 检查权重支持
        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        # 初始化损失函数
        if self.args.model_type in ["bertspan"]:
            self.loss_fct = None
        else:
            self.loss_fct = init_loss(
                weight=self.weight, device=self.device, args=self.args
            )

        # ONNX模型处理
        if self.args.onnx:
            self._initialize_onnx_model(model_name, use_cuda, onnx_execution_provider)
        else:
            self._initialize_pytorch_model(model_class, model_name, **kwargs)

        # 初始化分词器
        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        # 添加特殊token
        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _initialize_onnx_model(self, model_name, use_cuda, onnx_execution_provider):
        """初始化ONNX模型"""
        from onnxruntime import InferenceSession, SessionOptions

        if not onnx_execution_provider:
            onnx_execution_provider = (
                ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
            )

        options = SessionOptions()

        if self.args.dynamic_quantize:
            model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
            self.model = InferenceSession(
                model_path.as_posix(), options, providers=onnx_execution_provider
            )
        else:
            model_path = os.path.join(model_name, "onnx_model.onnx")
            self.model = InferenceSession(
                model_path, options, providers=onnx_execution_provider
            )

    def _initialize_pytorch_model(self, model_class, model_name, **kwargs):
        """初始化PyTorch模型"""
        quantized_weights = None
        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, **kwargs
            )
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )
            self.model = model_class.from_pretrained(
                None, config=self.config, state_dict=quantized_weights
            )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

    def _post_initialization_setup(self):
        """初始化后设置"""
        self.results = {}
        self.args.model_name = self.args.model_name if hasattr(self.args, 'model_name') else None
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

    # ====================
    # 训练相关方法
    # ====================

    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        训练模型

        Args:
            train_data: 训练数据
            output_dir: 输出目录
            show_running_loss: 是否显示运行损失
            args: 训练参数
            eval_data: 验证数据
            verbose: 是否详细输出

        Returns:
            global_step: 训练步数
            training_details: 训练详情
        """
        if args:
            self.args.update_from_dict(args)
        if self.args.silent:
            show_running_loss = False
        if not output_dir:
            output_dir = self.args.output_dir
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()
        train_dataset = self.load_and_cache_examples(train_data)
        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(model=self.model)
        logger.info(
            " Training of {} model complete. Saved to {}.".format(
                self.args.model_type, output_dir
            )
        )

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_data=None,
        test_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        执行模型训练

        Args:
            train_dataset: 训练数据集
            output_dir: 输出目录
            show_running_loss: 是否显示运行损失
            eval_data: 验证数据
            test_data: 测试数据
            verbose: 是否详细输出

        Returns:
            global_step: 训练步数
            training_details: 训练详情
        """
        model = self.model
        args = self.args

        # 设置TensorBoard
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

        # 创建数据加载器
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        # 计算总步数
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        # 设置优化器
        optimizer, scheduler = self._setup_optimizer_and_scheduler(model, t_total, args)

        # 多GPU设置
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # 训练状态初始化
        global_step, training_progress_scores = self._initialize_training_state(
            args, train_dataloader, output_dir, **kwargs
        )

        # 训练循环
        return self._execute_training_loop(
            model, args, train_dataloader, optimizer, scheduler, tb_writer,
            global_step, training_progress_scores, show_running_loss,
            eval_data, test_data, verbose, output_dir, **kwargs
        )

    def _setup_optimizer_and_scheduler(self, model, t_total, args):
        """设置优化器和学习率调度器"""
        # 参数分组
        optimizer_grouped_parameters = self._prepare_optimizer_parameters(model)

        # 计算预热步数
        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        # 初始化优化器
        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        # 初始化学习率调度器
        scheduler = self._get_scheduler(optimizer, args, t_total)

        return optimizer, scheduler

    def _prepare_optimizer_parameters(self, model):
        """准备优化器参数"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        custom_parameter_names = set()

        # 自定义参数组
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        # 自定义层参数
        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        # 默认参数
        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        return optimizer_grouped_parameters

    def _get_scheduler(self, optimizer, args, t_total):
        """获取学习率调度器"""
        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)
        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )
        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )
        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )
        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )
        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        return scheduler

    def _initialize_training_state(self, args, train_dataloader, output_dir, **kwargs):
        """初始化训练状态"""
        global_step = 0
        training_progress_scores = None
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # 检查是否从检查点继续训练
        if args.model_name and os.path.exists(args.model_name):
            global_step, epochs_trained, steps_trained_in_current_epoch = self._load_checkpoint(
                args, train_dataloader
            )

        # 初始化训练进度记录
        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        # 初始化wandb
        if args.wandb_project:
            self._init_wandb(args)

        return global_step, training_progress_scores

    def _load_checkpoint(self, args, train_dataloader):
        """加载检查点"""
        try:
            # 从模型路径获取global_step
            checkpoint_suffix = args.model_name.split("/")[-1].split("-")
            if len(checkpoint_suffix) > 2:
                checkpoint_suffix = checkpoint_suffix[1]
            else:
                checkpoint_suffix = checkpoint_suffix[-1]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            logger.info(
                "   Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("   Continuing training from epoch %d", epochs_trained)
            logger.info("   Continuing training from global step %d", global_step)
            logger.info(
                "   Will skip the first %d steps in the current epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("   Starting fine-tuning.")
            global_step = 0
            epochs_trained = 0
            steps_trained_in_current_epoch = 0

        return global_step, epochs_trained, steps_trained_in_current_epoch

    def _init_wandb(self, args):
        """初始化wandb"""
        wandb.init(
            project=args.wandb_project,
            config={**asdict(args)},
            **args.wandb_kwargs,
        )
        wandb.run._label(repo="nerpy")
        wandb.watch(self.model)
        self.wandb_run_id = wandb.run.id

    def _execute_training_loop(
        self, model, args, train_dataloader, optimizer, scheduler, tb_writer,
        global_step, training_progress_scores, show_running_loss,
        eval_data, test_data, verbose, output_dir, **kwargs
    ):
        """执行训练循环"""
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()

        # 初始化早停相关变量
        best_eval_metric = None
        early_stopping_counter = 0

        # FP16设置
        if self.args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        # 训练周期循环
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0

        for _ in train_iterator:
            model.train()
            # 处理从检查点继续训练的情况
            # ... (训练循环的具体实现)
            # 为简洁起见，此处省略具体实现，实际代码中会包含完整的训练循环逻辑

            epoch_number += 1

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    # ====================
    # 评估相关方法
    # ====================

    def eval_model(
        self,
        eval_data,
        output_dir=None,
        verbose=True,
        silent=False,
        wandb_log=True,
        **kwargs,
    ):
        """
        评估模型

        Args:
            eval_data: 评估数据
            output_dir: 输出目录
            verbose: 是否详细输出
            silent: 是否静默模式
            wandb_log: 是否记录到wandb

        Returns:
            result: 评估结果
            model_outputs: 模型输出
            preds_list: 预测结果列表
        """
        if not output_dir:
            output_dir = self.args.output_dir
        self._move_model_to_device()
        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)
        result, model_outputs, preds_list = self.evaluate(
            eval_dataset,
            output_dir,
            verbose=verbose,
            silent=silent,
            wandb_log=wandb_log,
            **kwargs,
        )
        self.results.update(result)
        if verbose:
            logger.debug(self.results)
        return result, model_outputs, preds_list

    def evaluate(
        self,
        eval_dataset,
        output_dir,
        verbose=True,
        silent=False,
        wandb_log=True,
        **kwargs,
    ):
        """
        执行模型评估

        Args:
            eval_dataset: 评估数据集
            output_dir: 输出目录
            verbose: 是否详细输出
            silent: 是否静默模式
            wandb_log: 是否记录到wandb

        Returns:
            results: 评估结果
            model_outputs: 模型输出
            preds_list: 预测结果列表
        """
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir
        results = {}
        id2label = {i: label for i, label in enumerate(self.args.labels_list)}
        span_metric = SpanEntityScore(id2label)

        # 创建数据加载器
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        out_input_ids = None
        out_attention_mask = None
        true_subjects = []
        model_outputs = []
        preds_list = []

        model.eval()
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        # 评估循环
        for batch in tqdm(
            eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )
                    tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # 根据模型类型处理输出
            if args.model_type in ["bertspan"]:
                self._process_span_model_output(
                    logits, batch, id2label, span_metric,
                    true_subjects, preds_list, model_outputs
                )
            else:
                self._process_sequence_model_output(
                    logits, inputs, preds, out_label_ids, out_input_ids,
                    out_attention_mask, id2label
                )

        # 计算最终结果
        return self._compute_evaluation_results(
            args, eval_loss, nb_eval_steps, id2label, span_metric,
            true_subjects, preds_list, model_outputs,
            preds, out_label_ids, out_input_ids, out_attention_mask,
            eval_output_dir, wandb_log, **kwargs
        )

    def _process_span_model_output(
        self, logits, batch, id2label, span_metric,
        true_subjects, preds_list, model_outputs
    ):
        """处理Span模型输出"""
        start_pred = torch.argmax(logits[0], -1).cpu().numpy()
        end_pred = torch.argmax(logits[1], -1).cpu().numpy()
        input_lens = batch[5].cpu().numpy()
        outputs = get_span_subject(start_pred, end_pred, input_lens)
        start_ids = batch[3].tolist()
        end_ids = batch[4].tolist()
        true_subject = get_span_subject(start_ids, end_ids)
        true_subjects.append(true_subject)
        for t, p in zip(true_subject, outputs):
            span_metric.update(true_subject=t, pred_subject=p)
        pred_entities = []
        for i in outputs:
            pred = []
            for x in i:
                pred.append([id2label[x[0]], x[1], x[2]])
            pred_entities.append(pred)
        preds_list.append(pred_entities)
        model_outputs.append(outputs)

    def _process_sequence_model_output(
        self, logits, inputs, preds, out_label_ids, out_input_ids,
        out_attention_mask, id2label
    ):
        """处理序列模型输出"""
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            out_input_ids = inputs["input_ids"].detach().cpu().numpy()
            out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
            out_input_ids = np.append(
                out_input_ids,
                inputs["input_ids"].detach().cpu().numpy(),
                axis=0,
            )
            out_attention_mask = np.append(
                out_attention_mask,
                inputs["attention_mask"].detach().cpu().numpy(),
                axis=0,
            )

    def _compute_evaluation_results(
        self, args, eval_loss, nb_eval_steps, id2label, span_metric,
        true_subjects, preds_list, model_outputs,
        preds, out_label_ids, out_input_ids, out_attention_mask,
        eval_output_dir, wandb_log, **kwargs
    ):
        """计算评估结果"""
        eval_loss = eval_loss / nb_eval_steps
        results = {}

        if args.model_type in ["bertspan"]:
            logger.debug(f"pred: {model_outputs[0]}")
            logger.debug(f"true: {true_subjects[0]}")
            eval_info, entity_info = span_metric.result()
            result = {"eval_loss": eval_loss}
            result.update(eval_info)
        else:
            token_logits = preds
            preds = np.argmax(preds, axis=2)
            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]
            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != self.pad_token_label_id:
                        out_label_list[i].append(id2label[out_label_ids[i][j]])
                        preds_list[i].append(id2label[preds[i][j]])
            word_tokens = []
            for i in range(len(preds_list)):
                w_log = self._convert_tokens_to_word_logits(
                    out_input_ids[i],
                    out_label_ids[i],
                    out_attention_mask[i],
                    token_logits[i],
                )
                word_tokens.append(w_log)
            model_outputs = [
                [word_tokens[i][j] for j in range(len(preds_list[i]))]
                for i in range(len(preds_list))
            ]
            extra_metrics = {}
            for metric, func in kwargs.items():
                if metric.startswith("prob_"):
                    extra_metrics[metric] = func(out_label_list, model_outputs)
                else:
                    extra_metrics[metric] = func(out_label_list, preds_list)
            result = {
                "eval_loss": eval_loss,
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1_score": f1_score(out_label_list, preds_list),
                **extra_metrics,
            }
            os.makedirs(eval_output_dir, exist_ok=True)
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "w", encoding="utf8") as writer:
                if args.classification_report:
                    cls_report = classification_report(
                        out_label_list, preds_list, digits=4
                    )
                    writer.write("{}\n".format(cls_report))
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))

            if self.args.wandb_project and wandb_log:
                self._log_to_wandb(args, out_label_list, preds_list, model_outputs)

        results.update(result)
        return results, model_outputs, preds_list

    def _log_to_wandb(self, args, out_label_list, preds_list, model_outputs):
        """记录到wandb"""
        wandb.init(
            project=args.wandb_project,
            config={**asdict(args)},
            **args.wandb_kwargs,
        )
        wandb.run._label(repo="nerpy")
        labels_list = sorted(self.args.labels_list)
        truth = [tag for out in out_label_list for tag in out]
        preds = [tag for pred_out in preds_list for tag in pred_out]
        outputs = [
            np.mean(logits, axis=0)
            for output in model_outputs
            for logits in output
        ]
        # ROC
        wandb.log({"roc": wandb.plots.ROC(truth, outputs, labels_list)})
        # Precision Recall
        wandb.log(
            {"pr": wandb.plots.precision_recall(truth, outputs, labels_list)}
        )
        # Confusion Matrix
        wandb.sklearn.plot_confusion_matrix(
            truth,
            preds,
            labels=labels_list,
        )

    # ====================
    # 预测相关方法
    # ====================

    def predict(self, to_predict, split_on_space=False):
        """
        执行预测

        Args:
            to_predict: 待预测文本列表
            split_on_space: 是否按空格分割

        Returns:
            preds: 预测结果
            model_outputs: 模型输出
            entities: 实体列表
        """
        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        id2label = {i: label for i, label in enumerate(self.args.labels_list)}
        preds = None
        span_preds = []
        model_outputs = []
        entities = []

        # 准备预测样本
        if split_on_space:
            predict_examples = [
                InputExample(
                    i,
                    sentence.split(),
                    [self.args.labels_list[0] for word in sentence.split()],
                )
                for i, sentence in enumerate(to_predict)
            ]
        else:
            predict_examples = [
                InputExample(
                    i, sentence, [self.args.labels_list[0] for word in sentence]
                )
                for i, sentence in enumerate(to_predict)
            ]

        # ONNX模型预测
        if self.args.onnx:
            return self._predict_with_onnx(
                to_predict, predict_examples, split_on_space, id2label
            )
        else:
            return self._predict_with_pytorch(
                predict_examples, device, model, args, pad_token_label_id,
                id2label, split_on_space, to_predict
            )

    def _predict_with_onnx(
        self, to_predict, predict_examples, split_on_space, id2label
    ):
        """使用ONNX模型进行预测"""
        # 实现ONNX预测逻辑
        pass

    def _predict_with_pytorch(
        self, predict_examples, device, model, args, pad_token_label_id,
        id2label, split_on_space, to_predict
    ):
        """使用PyTorch模型进行预测"""
        # 实现PyTorch预测逻辑
        pass

    # ====================
    # 辅助方法
    # ====================

    def _convert_tokens_to_word_logits(
        self, input_ids, label_ids, attention_mask, logits
    ):
        """
        将token级别的logits转换为word级别的logits
        """
        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]

        # 移除无用位置
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # 映射到词级别的logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, to_predict=None
    ):
        """
        加载并缓存样本

        Args:
            data: 数据路径或DataFrame
            evaluate: 是否为评估模式
            no_cache: 是否禁用缓存
            to_predict: 待预测样本

        Returns:
            dataset: 数据集
        """
        process_count = self.args.process_count
        tokenizer = self.tokenizer
        args = self.args
        if not no_cache:
            no_cache = args.no_cache
        mode = "dev" if evaluate else "train"

        # 使用HuggingFace数据集
        if self.args.use_hf_datasets and data:
            return self._load_hf_dataset(
                data, tokenizer, args, mode, process_count, no_cache
            )
        # 使用自定义数据集类
        elif args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(data, tokenizer, args, mode, to_predict)
        # 使用BERT Span数据集
        elif args.model_type in ["bertspan"]:
            return BertSpanDataset(data, tokenizer, args, mode, to_predict)
        # 使用默认处理方式
        else:
            return self._load_default_dataset(
                data, tokenizer, args, mode, process_count,
                no_cache, to_predict, evaluate
            )

    def _load_hf_dataset(
        self, data, tokenizer, args, mode, process_count, no_cache
    ):
        """加载HuggingFace数据集"""
        dataset = load_hf_dataset(
            data,
            self.args.labels_list,
            self.args.max_seq_length,
            self.tokenizer,
            # XLNet has a CLS token at the end
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token_id,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token_id,
            sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
            # PAD on the left for XLNet
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=self.pad_token_label_id,
            silent=args.silent,
            args=self.args,
        )
        return dataset

    def _load_default_dataset(
        self, data, tokenizer, args, mode, process_count,
        no_cache, to_predict, evaluate
    ):
        """加载默认数据集"""
        if not to_predict and isinstance(data, str) and self.args.lazy_loading:
            dataset = LazyNERDataset(data, tokenizer, self.args)
        else:
            examples = self._prepare_examples(data, to_predict, mode)
            features = self._convert_examples_to_features(
                examples, tokenizer, args, mode, process_count, no_cache
            )

            if self.args.onnx:
                return self._prepare_onnx_features(features)
            else:
                return self._prepare_tensor_dataset(features)

        return dataset

    def _prepare_examples(self, data, to_predict, mode):
        """准备样本"""
        if to_predict:
            examples = to_predict
        else:
            if isinstance(data, str):
                examples = read_examples_from_file(data, mode)
            else:
                if self.args.lazy_loading:
                    raise ValueError(
                        "Input must be given as a path to a file when using lazy loading"
                    )
                examples = [
                    InputExample(
                        guid=sentence_id,
                        words=sentence_df["words"].tolist(),
                        labels=sentence_df["labels"].tolist(),
                    )
                    for sentence_id, sentence_df in data.groupby(
                        ["sentence_id"]
                    )
                ]
        return examples

    def _convert_examples_to_features(
        self, examples, tokenizer, args, mode, process_count, no_cache
    ):
        """将样本转换为特征"""
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                args.model_type,
                args.max_seq_length,
                self.num_labels,
                len(examples),
            ),
        )
        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)
        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not no_cache)
            or (
                mode == "dev" and args.use_cached_eval_features and not no_cache
            )
        ):
            features = torch.load(cached_features_file)
            logger.info(
                f" Features loaded from cache at {cached_features_file}"
            )
        else:
            logger.info(" Converting to features started.")
            features = convert_examples_to_features(
                examples,
                self.args.labels_list,
                self.args.max_seq_length,
                self.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.pad_token_label_id,
                process_count=process_count,
                silent=args.silent,
                use_multiprocessing=args.use_multiprocessing,
                chunksize=args.multiprocessing_chunksize,
                mode=mode,
                use_multiprocessing_for_evaluation=args.use_multiprocessing_for_evaluation,
            )
            if not no_cache:
                torch.save(features, cached_features_file)

        return features

    def _prepare_onnx_features(self, features):
        """准备ONNX特征"""
        all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        return all_label_ids

    def _prepare_tensor_dataset(self, features):
        """准备Tensor数据集"""
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        return dataset

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """
        将模型转换为ONNX格式

        Args:
            output_dir: 输出目录
            set_onnx_arg: 是否设置ONNX参数
        """
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(
                    output_dir
                )
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")
        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)
            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="ner",
                opset=11,
            )
        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self._save_model_args(output_dir)

    def _calculate_loss(self, model, inputs, loss_fct, num_labels, args):
        """
        计算损失

        Returns:
            loss: 损失值
            outputs[1:]: 模型输出
        """
        outputs = model(**inputs)
        # model outputs are always tuple in pytorch-transformers (see doc)
        loss = outputs[0]
        if loss_fct:
            logits = outputs[1]
            labels = inputs["labels"]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss, outputs[1:]

    def _move_model_to_device(self):
        """将模型移动到指定设备"""
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        """获取最后的指标值"""
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        """
        获取输入字典

        Returns:
            inputs: 输入字典
        """
        if self.args.use_hf_datasets and isinstance(batch, dict):
            return {key: value.to(self.device) for key, value in batch.items()}
        else:
            batch = tuple(t.to(self.device) for t in batch)
            # 为BertSpan模型设置start_ids和end_ids
            if self.args.model_type in ["bertspan"]:
                # all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                if self.args.model_type in ["bert", "xlnet", "albert"]:
                    inputs["token_type_ids"] = batch[2]
            return inputs

    def _create_training_progress_scores(self, **kwargs):
        """创建训练进度分数记录"""
        return collections.defaultdict(list)

    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        """
        保存模型

        Args:
            output_dir: 输出目录
            optimizer: 优化器
            scheduler: 调度器
            model: 模型
            results: 结果
        """
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if model and not self.args.no_save:
            # 处理分布式/并行训练
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w", encoding="utf8") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        """保存模型参数"""
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        """加载模型参数"""
        args = NERArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        """获取命名参数"""
        return [n for n, p in self.model.named_parameters()]
