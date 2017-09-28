import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores

        # Best model -> with default value as n_constant
        best_n = self.n_constant

        # Minimize bic score; initialize to +infinity
        bic_score = math.inf

        try:
            for n in range(self.min_n_components, self.max_n_components+1):
                # BIC = -2 * logL + p * logN
                # logL -> log likelihood of fitted model
                # p -> number of parameters
                # N -> number of data points
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                p = n**2 + 2*n*model.n_features - 1
                score = 2*logL + p*(math.log(n))
                if score < bic_score:
                    best_n = n
                    bic_score = score
        except Exception as e:
            if self.verbose:
                print(e)
        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        # log(P(X(i))) -> log likelihood score of model on word i
        # log(P(X(all but i))) -> log likelihood score of model on all words other than word i
        # M -> total words

        # list of logL of all words
        log_likelihood_all = list()

        # best model with default value as n_constant
        best_n = self.n_constant

        # maximize dic score; initialize to -infinity
        dic_score = -math.inf

        try:
            for n in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(n)
                log_likelihood_all.append(model.score(self.X, self.lengths))

            M = self.max_n_components - self.min_n_components + 1
            logL_sum = sum(log_likelihood_all)

            for n in range(self.min_n_components, self.max_n_components+1):
                logL = log_likelihood_all[n - self.min_n_components]
                logL_others = logL_sum - logL
                score = logL - (logL_others / (M-1))
                if score > dic_score:
                    dic_score = score
                    best_n = n
        except Exception as e:
            if self.verbose:
                print(e)
        return self.base_model(best_n)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        mean_scores = []
        split_method = KFold()

        best_mean_score = -math.inf
        best_n = self.n_constant

        try:
            for n in range(self.min_n_components, self.max_n_components+1):
                # Fold and calculate model mean scores
                fold_scores = []
                for train_idx, test_idx in split_method.split(self.sequences):
                    # Training sequences
                    train_X, train_lengths = combine_sequences(train_idx, self.sequences)
                    # Test sequences
                    test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                    # Run model
                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    # Record each model score
                    fold_scores.append(hmm_model.score(test_X, test_lengths))
                # Compute mean of all fold scores
                mean_score = np.mean(fold_scores)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_n = n
        except Exception as e:
            if self.verbose:
                print(e)

        return self.base_model(best_n)
