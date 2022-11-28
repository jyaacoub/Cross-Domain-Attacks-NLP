"""
This is where we will run the application from and do tests
"""
from tqdm import tqdm

from textattack.attack_recipes import (
    A2TYoo2021,
    BAEGarg2019,
    DeepWordBugGao2018,
    PWWSRen2019,
    TextBuggerLi2018,
    TextFoolerJin2019,
)

from src.models.attack_model import AttackModel
from src.models.target_model import TargetModel

DOMAINS = {
    # David
    "similar_domain_same_task": {
        "attack_model": "MDL_IMDB_SENTIMENT",  # output_labels: 2
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    }, 
    # David
    "similar_domain_different_task": {
        "attack_model": "MDL_TWIT_IRONY",      # output_labels: 2
        "target_model": "MDL_TWIT_OFFENSIVE",  # output_labels: 3
        "target_dataset": "tweet_eval"
    },
    # Jean
    "different_domain_same_task": {
        "attack_model": "MDL_TWIT_OFFENSIVE",  # output_labels: 3
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    },
    # Jean
    "different_domain_different_task": {
        "attack_model": "MDL_TWIT_IRONY",      # output_labels: 2
        "target_model": "MDL_RT_SENTIMENT",    # output_labels: 2
        "target_dataset": "rotten_tomatoes"
    }
}

ATTACKS_RECIPES = [
    A2TYoo2021,
    BAEGarg2019,
    DeepWordBugGao2018,
    PWWSRen2019,
    TextBuggerLi2018,
    TextFoolerJin2019
]

def main():
    for domain_name in tqdm(DOMAINS, desc="Domains"):
        domain = DOMAINS[domain_name]
        for attack_recipe in tqdm(ATTACKS_RECIPES, desc='Attacks', leave=False):
            # "Substitute" model we are using to create adversarial examples with
            # a predefined attack recipe from TextAttack
            attack_model = AttackModel(
                model_name=domain["attack_model"],
                target_dataset=domain["target_dataset"],
                attack_recipe=attack_recipe,
                use_cuda=True
            )

            # We will generate 1,000 adversarial examples with the substitute model
            # and see how well they transfer to the other model in the other domain
            attack_results = attack_model.generate_target_examples(
                num_examples=1000, 
                query_budget=200,  # To reduce the running time
                log=True,
                dir=domain_name,
                # **{"parallel": True, "num_workers_per_device": 4}
            )

if __name__ == "__main__":
    main()