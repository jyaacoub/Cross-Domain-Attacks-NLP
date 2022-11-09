import pandas as pd
from tqdm import tqdm
from IPython.core.display import HTML, display

import torch
from torchinfo import summary

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import textattack
from textattack import Attack, Attacker, AttackArgs
from textattack.datasets import HuggingFaceDataset
from textattack.loggers import CSVLogger

from textattack.attack_recipes import (
    A2TYoo2021,
    BAEGarg2019,
    BERTAttackLi2020,
    # CLARE2020,
    CheckList2020,
    DeepWordBugGao2018,
    FasterGeneticAlgorithmJia2019,
    GeneticAlgorithmAlzantot2018,
    HotFlipEbrahimi2017,
    IGAWang2019,
    InputReductionFeng2018,
    Kuleshov2017,
    MorpheusTan2020,
    PSOZang2020,
    PWWSRen2019,
    Pruthi2019,
    Seq2SickCheng2018BlackBox,
    TextBuggerLi2018,
    TextFoolerJin2019 # best performing attack
 )

# transformers.logging.set_verbosity_error()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
