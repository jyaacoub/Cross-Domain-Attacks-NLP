from enum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [e.value for e in cls]


# Models:
class MODEL_PATHS(ExtendedEnum):
    ## Roberta-base models:
    # MOVIE REVIEWS DOMAIN (IMDB and Rotten Tomatoes)
    MDL_IMDB_SENTIMENT = (
        'textattack/roberta-base-imdb'  # Output: LABEL_0 (negative) LABEL_1 (positive)
    )
    MDL_RT_SENTIMENT = 'textattack/roberta-base-rotten-tomatoes'  # Output: LABEL_0 (negative) LABEL_1 (positive)

    # TWITTER DOMAIN
    MDL_TWIT_IRONY = 'cardiffnlp/bertweet-base-irony'          # Output: LABEL_0 (non-ironic) LABEL_1 (ironic)
    MDL_TWIT_OFFENSIVE = 'cardiffnlp/bertweet-base-offensive'  # Output: LABEL_0 (Not-offensive), LABEL_1 (Offensive)


# Datasets:
class DATASET_PATHS(ExtendedEnum):
    DATA_RT = 'rotten_tomatoes'
    DATA_IMDB = 'imdb'
    DATA_TWIT = 'tweet_eval'
