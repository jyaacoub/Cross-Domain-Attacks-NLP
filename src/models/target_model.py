import time

import torch
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.constants import DATASET_PATHS, MODEL_PATHS


class TargetModel:
    MODELS = MODEL_PATHS
    DATASETS = DATASET_PATHS

    def __init__(
        self,
        model_path=MODEL_PATHS.list()[0],
        use_cuda=False,
        dataset=DATASET_PATHS.list()[0],
    ):
        pass
