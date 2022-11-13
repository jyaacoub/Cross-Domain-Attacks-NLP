"""
Benchmark for the time and accuracy of the different predefined
attacks/recipes in TextAttack.
"""

import time

import pandas as pd
import torch
import transformers
from textattack import Attack, AttackArgs, Attacker
from textattack.attack_recipes import (
    CLARE2020,
    A2TYoo2021,
    BAEGarg2019,
    BERTAttackLi2020,
    CheckList2020,
    DeepWordBugGao2018,
    FasterGeneticAlgorithmJia2019,
    GeneticAlgorithmAlzantot2018,
    HotFlipEbrahimi2017,
    IGAWang2019,
    InputReductionFeng2018,
    Kuleshov2017,
    MorpheusTan2020,
    Pruthi2019,
    PSOZang2020,
    PWWSRen2019,
    Seq2SickCheng2018BlackBox,
    TextBuggerLi2018,
    TextFoolerJin2019,
)
from textattack.datasets import HuggingFaceDataset
from textattack.loggers import CSVLogger
from textattack.models.wrappers import HuggingFaceModelWrapper
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

transformers.logging.set_verbosity_error()

ATTACKS = [
    A2TYoo2021,
    BAEGarg2019,
    # BERTAttackLi2020,              # Takes too long, see https://github.com/QData/TextAttack/issues/586
    # CLARE2020,                     # Takes too long
    # CheckList2020,                 # Accuracy is not reduced with this attack
    DeepWordBugGao2018,
    # FasterGeneticAlgorithmJia2019, # Takes too long
    # GeneticAlgorithmAlzantot2018,  # Uses too much RAM
    # HotFlipEbrahimi2017,           # Cannot perform GradientBasedWordSwap on model
    # IGAWang2019,                   # Takes too long
    # InputReductionFeng2018,        # Accuracy is not reduced with this attack
    # Kuleshov2017,                  # Takes too long
    # MorpheusTan2020,               # Invalid text_input type <class 'torch.Tensor'> (required str or OrderedDict)
    # PSOZang2020,                   # Takes too long
    PWWSRen2019,
    # Pruthi2019,                    # Takes too long
    # Seq2SickCheng2018BlackBox,     # split() missing 1 required positional argument: 'split_size'
    TextBuggerLi2018,
    TextFoolerJin2019,
]
SUBSTITUTE_MODEL = "textattack/roberta-base-imdb"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    substitute_tokenizer = AutoTokenizer.from_pretrained(SUBSTITUTE_MODEL)
    substitute_model = AutoModelForSequenceClassification.from_pretrained(
        SUBSTITUTE_MODEL
    )
    # Move it to GPU, if possible
    substitute_model = substitute_model.to(device)

    # Wrap it for TextAttack
    model_wrapper = HuggingFaceModelWrapper(
        model=substitute_model, tokenizer=substitute_tokenizer
    )

    # Define the dataset we are going to pass to the model
    dataset = HuggingFaceDataset(
        name_or_dataset="rotten_tomatoes", subset=None, split="test", shuffle=False
    )

    benchmark = pd.DataFrame(
        columns=['Attack', 'Time', 'Original_Accuracy', 'Perturbed_accuracy']
    )

    for attack_class in tqdm(ATTACKS):
        attack = attack_class.build(model_wrapper)
        attack_args = AttackArgs(
            num_examples=100,
            random_seed=42,
            log_to_csv="log.csv",
            disable_stdout=True,
            silent=True,
            parallel=False,
        )
        attacker = Attacker(attack, dataset, attack_args)

        t1 = time.time()
        attack_results = attacker.attack_dataset()
        t2 = time.time()

        logs = pd.read_csv('log.csv')
        original_accuracy = (
            logs['ground_truth_output'] == logs['original_output']
        ).sum() / len(logs)
        perturbed_accuracy = (
            logs['ground_truth_output'] == logs['perturbed_output']
        ).sum() / len(logs)

        new_row = {
            'Attack': attack_class.__name__,
            'Time': round(t2 - t1, 2),
            'Original_Accuracy': original_accuracy,
            'Perturbed_accuracy': perturbed_accuracy,
        }
        benchmark = benchmark.append(new_row, ignore_index=True)
        benchmark.to_csv('benchmark.csv', index=False)


if __name__ == '__main__':
    main()
