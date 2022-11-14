"""
This is where we will run the application from and do tests
"""
from tqdm import tqdm

from textattack.attack_recipes import (
    A2TYoo2021,
    BAEGarg2019,
    DeepWordBugGao2018,
    PSOZang2020,
    PWWSRen2019,
    TextBuggerLi2018,
    TextFoolerJin2019,
)

from src.models.attack_model import AttackModel
from src.models.target_model import TargetModel

DOMAINS = {
    "similar_domain_same_task": {
        "attack_model": "MDL_IMDB_SENTIMENT",  # output_labels: 2
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    }, 

    "similar_domain_different_task": {
        "attack_model": "MDL_TWIT_IRONY",      # output_labels: 2
        "target_model": "MDL_TWIT_SENTIMENT",  # output_labels: 3
        "target_dataset": "tweet_eval"
    },

    "different_domain_same_task": {
        "attack_model": "MDL_TWIT_SENTIMENT",  # output_labels: 3
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    },

    "different_domain_different_task": {
        "attack_model": "MDL_TWIT_IRONY",      # output_labels: 2
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    }
}

ATTACKS_RECIPES = [
    A2TYoo2021,
    # BAEGarg2019,
    DeepWordBugGao2018,
    # PSOZang2020,
    # PWWSRen2019,
    # TextBuggerLi2018,
    # TextFoolerJin2019
]

def main():
    for domain_name in tqdm(DOMAINS, desc="Domains"):
        domain = DOMAINS[domain_name]
        for attack_recipe in tqdm(ATTACKS_RECIPES, desc='Attacks', leave=False):
            attack_model = AttackModel(
                model_name=domain["attack_model"],
                target_dataset=domain["target_dataset"],
                attack_recipe=attack_recipe,
                use_cuda=True
            )
            attack_results = attack_model.generate_target_examples(
                num_examples=10, 
                log=True,
                dir=domain_name
            )

if __name__ == "__main__":
    main()