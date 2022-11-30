"""
This is where we will run the application from and do tests
"""
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import pandas as pd

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
        "target_model": "MDL_TWIT_SENTIMENT",  # output_labels: 3
        "target_dataset": "tweet_eval"
    },
    # Jean
    "different_domain_same_task": {
        "attack_model": "MDL_TWIT_SENTIMENT",  # output_labels: 3
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

def main():
    results = pd.DataFrame()
    for domain_name in tqdm(DOMAINS):
        domain = DOMAINS[domain_name]
        
        target_model = TargetModel(
            model_name=domain["target_model"],
            use_cuda=True
        )
        
        path = f"logs/attacks/{domain_name}"
        logs = sorted([file for file in listdir(path) if isfile(join(path, file))])
        
        
        new_row = {'Domain': domain_name}
        for log_csv in logs:
            original_accuracy, perturbed_accuracy = target_model.evaluate_attack(join(path, log_csv), save_csv=True)
            
            new_row['Original accuracy'] = original_accuracy
            new_row[f'{log_csv.split("-")[0]} accuracy'] = perturbed_accuracy
            
        results = results.append(new_row, ignore_index=True)

    return results

if __name__ == "__main__":
    main()