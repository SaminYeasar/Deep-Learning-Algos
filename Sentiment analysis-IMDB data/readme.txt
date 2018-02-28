### Data set description
IMDB dataset :
* IMDB dataset consists of 15000 reviews in training set, 10000 reviews in validation set, and 25000 reviews in test set. This is a 2 class problem with class 1 being positive sentiment and class 0 being negative sentiment.

Preprocess of the data:
* Excluded all punctuation and converted everything into lower case
* Made a vocabulary of 10000 most common words from training data and discarded anything not found within the vocabulary.
* constructed bag of words, as in converted text into number that works as features.

Classifer used:
* As a baseline, report the performance of the random classiffier (a classiffier which classiffies a review into an uniformly random class). And then have done comparative studies using Naive Bayes, Decision Trees, and Linear SVM 