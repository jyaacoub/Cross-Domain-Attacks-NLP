import pandas as pd
import torch
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

from src.common.constants import DATASET_PATHS, MODEL_PATHS

logging.set_verbosity_error()


class AttackModel:
    """
    Loads attack models from a given huggingface model path into a wrapper class.
    and preps the model and dataset for attack.
    """

    MODELS = MODEL_PATHS
    DATASETS = DATASET_PATHS

    def __init__(
        self,
        model_name: str = "MDL_IMDB_SENTIMENT",
        target_dataset: str = 'rotten_tomatoes',
        use_cuda: bool = False,
        attack_recipe=TextFoolerJin2019,
    ):
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

        # wrapping model for textattack
        self.model_wrapped = HuggingFaceModelWrapper(self.model, self.model_tokenizer)

        # building attack and getting dataset
        self.attack_recipe = attack_recipe
        self.attack = attack_recipe.build(self.model_wrapped)
        # For the Twitter models we need the "sentiment" subset of the dataset
        self.subset = "sentiment" if target_dataset == "tweet_eval" else None
        self.target_dataset = self.set_target_dataset(target_dataset)
        self.attack_dataset = self.set_attack_dataset()

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

    def set_target_dataset(self, new_target: str):  # changes the target dataset
        """
        The target dataset is from the TargetModel domain.
        """
        return HuggingFaceDataset(  # dataset that is targeted by the attack
            name_or_dataset=new_target,
            subset=self.subset,
            split='test',
            shuffle=False,
        )

    def set_attack_dataset(self):
        """
        Maps the target dataset to the output of the AttackModel.

        That is, AttackModel will make predictions on the target dataset as an oracle and we
        will use these new labels to create the adversarial examples for AttackModel that we
        expect are going to transfer to the TargetModel.
        """
        original_dataset = self.target_dataset._dataset
        texts = tqdm(
            original_dataset['text'],
            leave=False,
            desc='Mapping target dataset to AttackModel',
        )
        attack_model_labels = [self.make_prediction(text) for text in texts]

        original_dataset = original_dataset.add_column(
            name="label_attack_model", column=attack_model_labels
        )

        return HuggingFaceDataset(
            name_or_dataset=original_dataset,
            dataset_columns=(["text"], "label_attack_model"),
            shuffle=False,
        )

    def generate_target_examples(
        self,
        num_examples: int = 10,
        query_budget: int = 200,
        log: bool = True,
        disable_stdout: bool = True,
        silent: bool = True,
        dir: str = "attacks",
        **kwargs,
    ):
        """
        This initiates the attack on the target domain by generating adversarial examples
        using our attack model and returns the results.

        Args:
            num_examples (`int`, optional): Number of examples to generate. Defaults to 10.
            log (`bool`, optional): Logs the examples to a file if True. Defaults to False.
            disable_stdout (`bool`, optional): Disable displaying individual attack results
                to stdout. Defaults to True.
            silent (`bool`, optional): Disable all logging (except for errors). Defaults to
                True.

        Returns:
            List[AttackResults]: returns a list of textattack.AttackResults containing the
                original and perturbed text as well as outputs
        """
        if log:
            log_to_csv = "logs/{}/{}-{}-{}.csv".format(
                dir,
                self.attack_recipe.__name__,
                self.model_path.split("/")[-1],
                self.target_dataset._name,
            )
        else:
            log_to_csv = None

        attack_args = AttackArgs(
            num_examples=num_examples,
            log_to_csv=log_to_csv,
            disable_stdout=disable_stdout,
            silent=silent,
            random_seed=42,
            query_budget=query_budget, # To reduce running time of the attacks
            **kwargs,
        )

        # NOTE: here it must be done with attack_dataset, not with target_dataset
        attack = Attacker(self.attack, self.attack_dataset, attack_args)
        attack_results = attack.attack_dataset()

        if log_to_csv:
            logs = pd.read_csv(log_to_csv)
            # Add the ground_truth of the original dataset to the logs, not of the attack dataset
            logs['ground_truth_output_target'] = self.target_dataset._dataset['label'][
                0 : len(logs)
            ]
            logs.to_csv(log_to_csv, index=False)

        return attack_results
