import torch
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.constants import MDL_TWIT_POLARITY  # huggingface model paths
from common.constants import MDL_IMDB_POLARITY, MDL_RT_POLARITY, MDL_TWIT_IRONY
import time


class AttackModel:
    """
    Loads attack models from a given huggingface model path into a wrapper class.
    and preps the model and dataset for attack.
    """
    # Models to load as static variables for easy access:
    IMDB_POLARITY = MDL_IMDB_POLARITY
    RT_POLARITY = MDL_RT_POLARITY # Rotten Tomatoes
    TWIT_POLARITY = MDL_TWIT_POLARITY
    TWIT_IRONY = MDL_TWIT_IRONY
    
    def __init__(self, model_path=MDL_IMDB_POLARITY, use_cuda=False, 
                 attack_recipe=TextFoolerJin2019, target_dataset='rotten_tomatoes'):
        self.model_path = model_path
        
        # extracting model and tokenizer from model path::
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Move model to GPU if available:
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.model.to(self.device)
        
        # wrapping model for textattack
        self.model_wrapped = HuggingFaceModelWrapper(self.model, self.model_tokenizer)
        
        # building attack and getting dataset
        self.attack = attack_recipe.build(self.model_wrapped)
        self.target_dataset = self.set_target_dataset(target_dataset)
    
    def set_target_dataset(self, new_target: str): # changes the target dataset
        self.target_dataset = HuggingFaceDataset( # dataset that is targeted by the attack
            name_or_dataset=new_target,
            subset=None,
            split='test',
            shuffle=False)
        return self.target_dataset
    
    def generate_target_examples(self, num_examples=10, log=False, 
                     disable_stdout=True, silent=True, **kwargs):
        """
        This initiates the attack on the target domain by generating adversarial examples 
        using our attack model.

        Args:
            num_examples (`int`, optional): Number of examples to generate. Defaults to 10.
            log (`bool`, optional): Logs the examples to a file if True. Defaults to False.
            disable_stdout (`bool`, optional): Disable displaying individual attack results to stdout. Defaults to True.
            silent (`bool`, optional): Disable all logging (except for errors). Defaults to True.

        Returns:
            List[AttackResults]: returns a list of textattack.AttackResults containing the original and perturbed text as well as outputs
        """
        
        if log:
            log_to_csv = "attacks/{}-{}-{}.csv".format(self.model_path.split("/")[-1], 
                                                       self.target_dataset._name,
                                                       time.strftime("%m%d_%H%M"))
        else:
            log_to_csv = None
            
        
        attack_args = AttackArgs(
            num_examples=num_examples,
            log_to_csv=log_to_csv, 
            disable_stdout=disable_stdout, 
            silent=silent,
            **kwargs
        )
        
        attack = Attacker(self.attack, self.target_dataset, attack_args)
        return attack.attack_dataset()
        
        
    
    
            
        