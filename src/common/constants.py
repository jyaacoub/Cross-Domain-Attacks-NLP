from enum import Enum
class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [e.value for e in cls]

# Models:
class MODEL_PATHS(ExtendedEnum):
    ## Roberta-base models:
    # MOVIE REVIEWS DOMAIN (IMDB and Rotten Tomatoes)
    MDL_IMDB_POLARITY = 'textattack/roberta-base-imdb' # Output: LABEL_0 (negative) LABEL_1 (positive)
    MDL_RT_POLARITY = 'textattack/roberta-base-rotten-tomatoes' # Output: LABEL_0 (negative) LABEL_1 (positive)


    # TWITTER DOMAIN
    MDL_TWIT_IRONY = 'cardiffnlp/twitter-roberta-base-irony' # Output: LABEL_0 (non-ironic) LABEL_1 (ironic)
    MDL_TWIT_POLARITY = 'cardiffnlp/twitter-roberta-base-sentiment' # Output: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)

# Datasets:
class DATASET_PATHS(ExtendedEnum):
    DATA_RT = 'rotten_tomatoes'
    DATA_IMDB = 'imdb'
    DATA_TWIT = 'twitter'