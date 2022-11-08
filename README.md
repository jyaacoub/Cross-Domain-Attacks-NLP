# Cross-Domain-Attacks-NLP

# File Structure:

* **media/** - relevent figures and results here
* **src/**
  * **common/**
    * `constants.py` - For constant variables used throughout the application.
  * **models/**
    * `adversarial_model.py` - Initial file for handling the trained adversarial model (this will include multiple models for testing)
    * `black_box_model.py` - File containing the black-box model for which we will be generating adversarial examples for.
  * **utils/** - any relevent util functions/classes here.
  * **data/** - any relevent datasets here.


* `run.py` - where we will run the application and do tests from.


# Testing Procedure
This section shows what models we will be testing as we build up to answering our research question.
## 0. Same Domain Same Task:
This has already been shown to work in many previous papers. No need to do anything for this.

## 1. Similar Domain, Same Task:
Here we replicate how [Datta S. 2022](https://arxiv.org/abs/2205.07315) tested similar domains by examining how adversarial examples for one category of amazon reviews transfers to another catagory (e.g.: from baby items to books). Here our similar domains are movie reviews, but from IMBD and from Rotten Tomatoes
* Similar domain being movie reviews but from different sites (*might not be different enough*)
  * As the first model we can use [textattack/roberta-base-imdb](https://huggingface.co/textattack/roberta-base-imdb)
  * As the second model we can use [textattack/roberta-base-rotten-tomatoes](https://huggingface.co/textattack/roberta-base-rotten-tomatoes)

## 2. Similar Domain, Different Task:
Building off of 1 we now try to test how adversarial examples transfer across different tasks (e.g.: sentiment analysis of polarity vs subjectivity).
* Model x vs Model Y
  * model x is ...
  * model y is ...
* For SAME DOMAIN we can test sentiment vs Irony with RoBERTa in Tweets (only difference in arch is output layer): (https://arxiv.org/pdf/2010.12421.pdf)
  * First model (irony, *out-2*): [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)
  * Second model (polarity, *out-3*): [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)

## 3. Different Domain, Same Task:
Now we test how models transfer across different domains but on the same task (e.g.: polarity of amazon reviews vs polarity of tweets; *tweets might not be different enough...*).
* Using the models from #2 and #3 with roberta-base and sentiment analysis (only difference in arch is output layer)
  * First model (Twitter, *out-3*): [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
  * Second model (RT, *out-2*): [textattack/roberta-base-rotten-tomatoes](https://huggingface.co/textattack/roberta-base-rotten-tomatoes)

## 4. Different Domain, Different Task:
Finally we arrive at the untimate black-box condition, if we get this far then this tells us a lot about where adversarial examples in NLP come from. It implies that they are a result of low level features of text? it also implies that models across domains have decision boundaries that are very similar.
* Here we also use RoBERTa. Domains are Rotten Tomatoes and Twitter, and tasks are polarity and irony (no difference in arch)
  * First model (irony, *out-2*): [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)
  * Second model (RT, *out-2*): [textattack/roberta-base-rotten-tomatoes](https://huggingface.co/textattack/roberta-base-rotten-tomatoes)

# References
- [Cross-Domain Transferability of Adversarial Perturbations](https://arxiv.org/abs/1905.11736)
- [Learn2Weight: Parameter Adaptation against Similar-domain Adversarial Attacks](https://arxiv.org/abs/2205.07315)
- [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909)
- [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421)
