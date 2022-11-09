# Models:

## Roberta-base models:
# MOVIE REVIEWS DOMAIN (IMDB and Rotten Tomatoes)
MDL_IMDB_POLARITY = 'textattack/roberta-base-imdb' # Output: LABEL_0 (negative) LABEL_1 (positive)
MDL_RT_POLARITY = 'textattack/roberta-base-rotten-tomatoes' # Output: LABEL_0 (negative) LABEL_1 (positive)


# TWITTER DOMAIN
MDL_TWIT_IRONY = 'cardiffnlp/twitter-roberta-base-irony' # Output: LABEL_0 (non-ironic) LABEL_1 (ironic)
MDL_TWIT_POLARITY = 'cardiffnlp/twitter-roberta-base-sentiment' # Output: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)


