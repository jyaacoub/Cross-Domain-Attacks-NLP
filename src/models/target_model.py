import time

import pandas as pd
import torch
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

from src.common.constants import DATASET_PATHS, MODEL_PATHS

logging.set_verbosity_error()


class TargetModel:
    MODELS = MODEL_PATHS
    DATASETS = DATASET_PATHS

    def __init__(self, model_name: str = "MDL_IMDB_SENTIMENT", use_cuda: bool = False):
        self.model_path = self.MODELS[model_name].value

        # extracting model and tokenizer from model path::
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Move model to GPU if available:
        self.device = (
            torch.device("cuda")
            if (use_cuda and torch.cuda.is_available())
            else torch.device("cpu")
        )
        self.model.to(self.device)

    # TODO: create a function that takes batches of texts
    def make_prediction(self, text: str) -> int:
        """
        Makes a prediction for a single text/sentence.
        """
        encoded_input = self.model_tokenizer(text, return_tensors='pt')
        for (k, tensor) in encoded_input.items():
            encoded_input[k] = tensor.to(self.device)

        prediction = self.model(**encoded_input).logits.softmax(dim=1).argmax().item()

        return prediction

    def evaluate_attack(self, log_csv: str):
        """
        Evaluates the attacks created with AttackModel that were stored in log_csv.
        """
        logs = pd.read_csv(log_csv)
        # TextAttack adds "[[ ]]" to the words/tokens that were changed, so we have to remove them
        logs["original_text"] = logs["original_text"].apply(
            lambda x: x.replace("[[", "").replace("]]", "")
        )
        logs["perturbed_text"] = logs["perturbed_text"].apply(
            lambda x: x.replace("[[", "").replace("]]", "")
        )

        logs['original_output'] = logs["original_text"].apply(self.make_prediction)
        logs['perturbed_output'] = logs["perturbed_text"].apply(self.make_prediction)

        # NOTE: be careful with the ground_truth_output, since in some cases the task will be different
        original_accuracy = (
            logs['ground_truth_output_target'] == logs['original_output']
        ).mean()
        perturbed_accuracy = (
            logs['ground_truth_output_target'] == logs['perturbed_output']
        ).mean()

        return original_accuracy, perturbed_accuracy
